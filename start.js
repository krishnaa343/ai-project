/**
 * Single-command starter for AI-Based Crop Disease Detection Framework.
 * Starts both the Flask backend (port 5000) and the React/Vite frontend (port 5173).
 * If the backend is already running, it smoothly connects and launches the frontend.
 */

const { spawn } = require('child_process');
const path = require('path');
const net = require('net');

const rootDir = __dirname;
const backendDir = path.join(rootDir, 'backend');
const frontendDir = path.join(rootDir, 'frontend');

const rawPythonExe = process.platform === 'win32'
  ? path.join(backendDir, 'venv', 'Scripts', 'python.exe')
  : path.join(backendDir, 'venv', 'bin', 'python');

const pythonExe = `"${rawPythonExe}"`;

console.log('====================================================');
console.log(' Starting Crop Disease Detection Framework');
console.log(' Backend: Flask (http://127.0.0.1:5000)');
console.log(' Frontend: Vite (http://localhost:5173)');
console.log(' Press Ctrl+C at any time to stop');
console.log('====================================================\n');

let backendProcess = null;
let frontendProcess = null;

// Helper to check if a port is in use
function isPortInUse(port, host = '127.0.0.1') {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    socket.setTimeout(800);
    socket.once('connect', () => {
      socket.destroy();
      resolve(true);
    });
    socket.once('timeout', () => {
      socket.destroy();
      resolve(false);
    });
    socket.once('error', () => {
      resolve(false);
    });
    socket.connect(port, host);
  });
}

async function main() {
  const backendRunning = await isPortInUse(5000);

  if (backendRunning) {
    console.log('[Backend] Notice: Detected existing backend server already listening on port 5000.');
  } else {
    console.log('[Backend] Launching Flask server on http://127.0.0.1:5000 ...');
    backendProcess = spawn(pythonExe, ['run.py'], {
      cwd: backendDir,
      shell: true,
      windowsVerbatimArguments: true,
      stdio: ['inherit', 'pipe', 'pipe']
    });

    backendProcess.stdout.on('data', (data) => {
      process.stdout.write(`[Backend] ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
      process.stderr.write(`[Backend] ${data}`);
    });

    backendProcess.on('error', (err) => {
      console.error('[Backend Error]:', err.message);
    });
  }

  // 2. Start Frontend
  console.log('[Frontend] Launching React/Vite development server on http://localhost:5173 ...');
  const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  frontendProcess = spawn(npmCmd, ['run', 'dev'], {
    cwd: frontendDir,
    shell: true,
    stdio: ['inherit', 'pipe', 'pipe']
  });

  frontendProcess.stdout.on('data', (data) => {
    process.stdout.write(`[Frontend] ${data}`);
  });

  frontendProcess.stderr.on('data', (data) => {
    process.stderr.write(`[Frontend] ${data}`);
  });

  frontendProcess.on('error', (err) => {
    console.error('[Frontend Error]:', err.message);
  });
}

// Graceful shutdown handling
function cleanup() {
  console.log('\nStopping servers...');
  try {
    if (backendProcess && backendProcess.pid) {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', backendProcess.pid, '/f', '/t'], { shell: true });
      } else {
        backendProcess.kill('SIGTERM');
      }
    }
  } catch (e) {}

  try {
    if (frontendProcess && frontendProcess.pid) {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', frontendProcess.pid, '/f', '/t'], { shell: true });
      } else {
        frontendProcess.kill('SIGTERM');
      }
    }
  } catch (e) {}

  setTimeout(() => process.exit(0), 1000);
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

main();
