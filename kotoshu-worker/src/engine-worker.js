// The worker entry (plan 112): hand this module to the Worker
// constructor and every engine call stays off the main thread.
//
//   import EngineWorker from '@kotoshu/worker/engine-worker'
//   const worker = new EngineWorker()   // vite/webpack resolve the URL
//
// or with the classic pattern:
//
//   new Worker(
//     new URL('@kotoshu/worker/engine-worker', import.meta.url),
//     { type: 'module' },
//   )
//
// The wiring is the whole file: an engine speaking the playground
// protocol (see engine-core.js) whose events become postMessage calls.
// The WorkerGlobalScope guard keeps the module importable outside a
// worker (Node, tests) without touching a nonexistent self.

import { createEngine } from './engine-core.js'

if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  const engine = createEngine((message) => {
    self.postMessage(message)
  })
  self.onmessage = (event) => {
    void engine.handle(event.data)
  }
}
