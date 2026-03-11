'use strict'

// Wrapper around the auto-generated all.js. Two mechanisms:
// 1. Keepalive: prevents bare from exiting prematurely during C++ model loading
//    (background threads run while JS has no pending work).
// 2. Exit detection: native addon handles keep the event loop alive after tests
//    finish, so we write .exit-code and let the shell/runner kill us.
const fs = require('bare-fs')
const path = require('bare-path')

const keepalive = setInterval(() => {}, 30_000)
const origLog = console.log
const EXIT_CODE_FILE = path.join(__dirname, '.exit-code')

function writeExitCode () {
  let code = 0
  if (global.Bare && global.Bare.exitCode !== undefined) code = global.Bare.exitCode
  try {
    fs.writeFileSync(EXIT_CODE_FILE, String(code))
  } catch (_) {}
}

let done = false
let stabilityPollRef = null
function onComplete () {
  if (done) return
  done = true
  clearInterval(keepalive)
  if (stabilityPollRef) clearInterval(stabilityPollRef)
  writeExitCode()
  const runner = global[Symbol.for('brittle-runner')]
  if (runner && typeof runner.end === 'function') {
    try { runner.end() } catch (_) {}
  }
}

console.log = function (...args) {
  origLog.apply(console, args)
  if (args.length === 1 && (args[0] === '# ok' || args[0] === '# not ok')) {
    onComplete()
  }
}

require('./all.js')

// Backup: poll test count stability (e.g. if console.log interception fails on Windows)
const RUNNER = Symbol.for('brittle-runner')
let lastCount = 0
let stableTicks = 0
stabilityPollRef = setInterval(function () {
  const runner = global[RUNNER]
  if (!runner || !runner.started) return

  const count = runner.tests.count
  if (count > 0 && count === lastCount && runner.next === null) {
    stableTicks++
  } else {
    stableTicks = 0
  }
  lastCount = count

  if (stableTicks >= 3) {
    onComplete()
  }
}, 5000)
