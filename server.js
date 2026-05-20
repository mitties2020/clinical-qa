const http = require('http');
const express = require('express');
const cors = require('cors');
const twilio = require('twilio');
const { WebSocketServer } = require('ws');
const { createClient: createDeepgramClient, LiveTranscriptionEvents } = require('@deepgram/sdk');
const { URL } = require('url');

const PORT = process.env.PORT || 10000;
const BASE_URL = process.env.BASE_URL;
const STREAM_URL = process.env.STREAM_URL || (BASE_URL ? BASE_URL.replace(/^http/, 'ws') + '/twilio-stream' : undefined);
const DOCTOR_PHONE = process.env.DOCTOR_PHONE;
const TWILIO_NUMBER = process.env.TWILIO_NUMBER;

const app = express();
app.use(cors());
app.use(express.json());

// Preserve existing routes/functionality by mounting an existing Express app/router when present.
try {
  const existing = require('./backend');
  const existingApp = existing?.default || existing?.app || existing;
  if (typeof existingApp === 'function') {
    app.use(existingApp);
    console.log('Mounted existing backend routes from ./backend');
  }
} catch (_) {
  // No-op when no separate backend module exists in this repo.
}

function requireEnv(name) {
  const value = process.env[name];
  if (!value) throw new Error(`Missing environment variable: ${name}`);
  return value;
}

function createTwilioClient() {
  return twilio(requireEnv('TWILIO_ACCOUNT_SID'), requireEnv('TWILIO_AUTH_TOKEN'));
}

function twimlConnectPatientResponse() {
  const response = new twilio.twiml.VoiceResponse();
  const connect = response.connect();
  connect.stream({ url: requireEnv('STREAM_URL') });
  return response.toString();
}

app.post('/api/call-patient', async (req, res) => {
  try {
    const { patientPhone } = req.body || {};
    if (!patientPhone) return res.status(400).json({ error: 'patientPhone is required in request body' });

    const client = createTwilioClient();
    const twimlUrl = new URL('/twiml/connect-patient', requireEnv('BASE_URL')).toString();

    const call = await client.calls.create({
      to: patientPhone,
      from: requireEnv('TWILIO_NUMBER'),
      url: twimlUrl,
      statusCallbackEvent: ['initiated', 'ringing', 'answered', 'completed'],
      statusCallback: new URL('/api/call-status', requireEnv('BASE_URL')).toString(),
      statusCallbackMethod: 'POST',
    });

    if (DOCTOR_PHONE) {
      await client.calls.create({
        to: DOCTOR_PHONE,
        from: requireEnv('TWILIO_NUMBER'),
        url: twimlUrl,
      });
    }

    res.json({ ok: true, sid: call.sid });
  } catch (error) {
    console.error('call-patient error:', error);
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/twiml/connect-patient', (_req, res) => {
  try {
    const xml = twimlConnectPatientResponse();
    res.type('text/xml').send(xml);
  } catch (error) {
    res.status(500).type('text/plain').send(error.message);
  }
});

app.post('/api/call-status', (req, res) => {
  console.log('Twilio call status:', req.body);
  res.sendStatus(204);
});

const server = http.createServer(app);

// Frontend transcript websocket
const frontendWss = new WebSocketServer({ noServer: true });
const frontendClients = new Set();
let transcriptionActive = false;

function broadcast(payload) {
  const message = JSON.stringify(payload);
  for (const client of frontendClients) {
    if (client.readyState === 1) client.send(message);
  }
}

function setTranscriptionActive(active) {
  transcriptionActive = !!active;
  broadcast({ type: 'status', status: transcriptionActive ? 'active' : 'stopped' });
}

function broadcastTranscript(text) {
  const transcript = String(text || '').trim();
  if (!transcript) return;
  broadcast({ type: 'transcript', text: transcript });
}

frontendWss.on('connection', (socket) => {
  frontendClients.add(socket);
  socket.send(JSON.stringify({ type: 'status', status: 'connected' }));
  socket.send(JSON.stringify({ type: 'status', status: transcriptionActive ? 'active' : 'stopped' }));
  socket.on('close', () => frontendClients.delete(socket));
});

// Twilio media stream websocket + Deepgram bridge
const twilioWss = new WebSocketServer({ noServer: true });

twilioWss.on('connection', (socket) => {
  let deepgramConn;

  try {
    const deepgram = createDeepgramClient(requireEnv('DEEPGRAM_API_KEY'));
    deepgramConn = deepgram.listen.live({
      model: 'nova-2',
      language: 'en',
      encoding: 'mulaw',
      sample_rate: 8000,
      channels: 1,
      interim_results: false,
      punctuate: true,
      smart_format: true,
    });

    deepgramConn.on(LiveTranscriptionEvents.Open, () => {
      setTranscriptionActive(true);
    });

    deepgramConn.on(LiveTranscriptionEvents.Transcript, (data) => {
      const transcript = data?.channel?.alternatives?.[0]?.transcript;
      if (transcript) broadcastTranscript(transcript);
    });

    deepgramConn.on(LiveTranscriptionEvents.Error, (err) => {
      console.error('Deepgram error:', err);
    });
  } catch (error) {
    console.error('Failed to initialize Deepgram:', error);
  }

  socket.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      if (msg.event === 'media' && msg.media?.payload && deepgramConn?.send) {
        deepgramConn.send(Buffer.from(msg.media.payload, 'base64'));
      }
      if (msg.event === 'stop') {
        setTranscriptionActive(false);
      }
    } catch (err) {
      console.error('Twilio stream message error:', err);
    }
  });

  socket.on('close', () => {
    setTranscriptionActive(false);
    if (deepgramConn?.finish) deepgramConn.finish();
  });
});

server.on('upgrade', (request, socket, head) => {
  let pathname;
  try {
    pathname = new URL(request.url, 'http://localhost').pathname;
  } catch {
    socket.destroy();
    return;
  }

  if (pathname === '/frontend-transcript') {
    frontendWss.handleUpgrade(request, socket, head, (ws) => frontendWss.emit('connection', ws, request));
    return;
  }
  if (pathname === '/twilio-stream') {
    twilioWss.handleUpgrade(request, socket, head, (ws) => twilioWss.emit('connection', ws, request));
    return;
  }
  socket.destroy();
});

server.listen(PORT, () => {
  console.log(`Server listening on ${PORT}`);
  console.log(`Frontend transcript WS: /frontend-transcript`);
  console.log(`Twilio stream WS: /twilio-stream`);
});
