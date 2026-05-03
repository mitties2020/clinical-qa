/**
 * Transcript websocket broadcast helper for an existing Node/Express + Twilio/Deepgram server.
 *
 * Usage in your current backend:
 *   const { attachFrontendTranscriptServer } = require("./transcript_bridge");
 *   const transcriptBridge = attachFrontendTranscriptServer(httpServer);
 *
 *   // Inside your Deepgram transcript callback:
 *   transcriptBridge.broadcastTranscript(transcript);
 *
 *   // On call lifecycle updates:
 *   transcriptBridge.setTranscriptionActive(true);  // when stream starts
 *   transcriptBridge.setTranscriptionActive(false); // when stream stops
 */
const { WebSocketServer } = require("ws");

function attachFrontendTranscriptServer(httpServer, path = "/frontend-transcript") {
  const wss = new WebSocketServer({ server: httpServer, path });
  let transcriptionActive = false;

  function broadcast(payload) {
    const message = JSON.stringify(payload);
    for (const client of wss.clients) {
      if (client.readyState === 1) client.send(message);
    }
  }

  function setTranscriptionActive(active) {
    transcriptionActive = Boolean(active);
    broadcast({ type: "status", status: transcriptionActive ? "active" : "stopped" });
  }

  function broadcastTranscript(transcript) {
    const text = String(transcript || "").trim();
    if (!text) return;
    broadcast({ type: "transcript", text });
  }

  wss.on("connection", (socket) => {
    socket.send(JSON.stringify({ type: "status", status: "connected" }));
    socket.send(JSON.stringify({ type: "status", status: transcriptionActive ? "active" : "stopped" }));
  });

  return { broadcastTranscript, setTranscriptionActive };
}

module.exports = { attachFrontendTranscriptServer };
