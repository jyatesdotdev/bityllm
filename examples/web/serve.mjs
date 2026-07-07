// Tiny static server for the web demo (zero deps).
//   node examples/web/serve.mjs   →  http://localhost:8143/examples/web/

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const PORT = 8143;
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css",
  ".json": "application/json",
  ".bity": "application/octet-stream",
};

createServer(async (req, res) => {
  try {
    let path = decodeURIComponent((req.url ?? "/").split("?")[0]);
    if (path.endsWith("/")) path += "index.html";
    const file = join(ROOT, path);
    if (!file.startsWith(ROOT)) throw new Error("nope");
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("404");
  }
}).listen(PORT, () => console.log(`bity terminal → http://localhost:${PORT}/examples/web/`));
