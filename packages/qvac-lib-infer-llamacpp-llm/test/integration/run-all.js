'use strict'

// Wrapper around the auto-generated all.js that keeps the bare event loop
// alive during native model loading. Without this, bare exits prematurely
// when C++ model loading runs in background threads while JS has no pending work.
// Clear the interval when brittle prints the final summary so the process can exit.
const keepalive = setInterval(() => {}, 30_000)
const origLog = console.log
console.log = function (...args) {
  origLog.apply(console, args)
  if (args.length === 1 && (args[0] === '# ok' || args[0] === '# not ok')) {
    clearInterval(keepalive)
    const runner = global[Symbol.for('brittle-runner')]
    if (runner && typeof runner.end === 'function') {
      try { runner.end() } catch (_) {}
    }
  }
}

require('./all.js')
