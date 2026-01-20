# WEBSOCKET ENDPOINT FOR REAL-TIME STREAMING
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection streaming.
    Protocol:
    1. Client connects to ws://server:8000/ws
    2. Server accepts connection
    3. Server continuously sends detection data as JSON
    4. Client receives updates in real-time
    Usage (JavaScript):
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Detections:', data.detections);
        console.log('Statistics:', data.statistics);
    };
    ```
    Args:
        websocket: FastAPI WebSocket connection
    """
    if pipeline is None:
        await websocket.close(code=1011, reason="Pipeline not initialized")
        return
    # Accept and register connection
    await pipeline.ws_manager.connect(websocket)
    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages from client (e.g., commands, config changes)
            data = await websocket.receive_text()
            # Echo back or process commands
            # For now, just acknowledge
            await websocket.send_json({
                "type": "ack",
                "message": "Message received",
                "data": data
            })
    except WebSocketDisconnect:
        # Client disconnected
        pipeline.ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        pipeline.ws_manager.disconnect(websocket)
# ENDPOINT FOR WEBSOCKET TESTING
@app.get("/test-ws", response_class=HTMLResponse)
async def test_websocket_page():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>WebSocket Tester</title>
        <style>body { font-family: Arial; margin: 20px; } #log { white-space: pre-wrap; background: #eee; padding: 10px; border-radius: 8px; }</style>
    </head>
    <body>
        <h2>WebSocket Test Page</h2>
        <p>Připojuji se na: <b>wss://192.168.37.205:8000/ws/ai</b></p>
        <button onclick="sendTest()">Send Test Message</button>
        <h3>Log:</h3>
        <div id="log"></div>
        <script>
            let logDiv = document.getElementById("log");
            function log(msg) {
                logDiv.textContent += msg + "\\n";
            }
            // Připojení na WebSocket server
            let ws = new WebSocket("wss://192.168.37.205:8000/ws/ai");
            ws.onopen = () => {
                log("WebSocket připojen!");
            };
            ws.onmessage = (event) => {
                log("Přišla zpráva: " + event.data);
            };
            ws.onerror = (error) => {
                log("Chyba: " + error);
            };
            ws.onclose = () => {
                log("WebSocket byl uzavřen.");
            };
            function sendTest() {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ test: "Hello from browser!" }));
                    log("Odesláno: Hello from browser!");
                } else {
                    log("WebSocket není připojen!");
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
