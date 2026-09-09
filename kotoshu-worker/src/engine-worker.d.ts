/**
 * @file The worker entry module — a side-effect module: it wires
 * createEngine to self.onmessage/postMessage when evaluated inside a
 * Worker. Hand it to new Worker() (see README.md); it exports nothing.
 */
