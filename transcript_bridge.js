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
const { URL } = require("url");

function attachFrontendTranscriptServer(httpServer, path = "/frontend-transcript") {
  const wss = new WebSocketServer({ noServer: true });
  const frontendClients = new Set();
  let transcriptionActive = false;

  function broadcast(payload) {
    const message = JSON.stringify(payload);
    for (const client of frontendClients) {
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
    console.log("Frontend transcript client connected");
    frontendClients.add(socket);
    socket.send(JSON.stringify({ type: "status", status: "connected" }));
    socket.send(JSON.stringify({ type: "status", status: transcriptionActive ? "active" : "stopped" }));
    socket.on("close", () => {
      frontendClients.delete(socket);
    });
  });

  // Important: this routes frontend transcript upgrades explicitly so
  // existing /twilio-stream upgrade handling can remain untouched.
  httpServer.on("upgrade", (request, socket, head) => {
    let pathname = "";
    try {
      pathname = new URL(request.url, "http://localhost").pathname;
    } catch {
      socket.destroy();
      return;
    }
    if (pathname !== path) return;

    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit("connection", ws, request);
    });
  });

  return { broadcastTranscript, setTranscriptionActive, frontendClients };
}

module.exports = { attachFrontendTranscriptServer };
