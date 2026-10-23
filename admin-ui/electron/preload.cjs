const { contextBridge, ipcRenderer } = require("electron");

/**
 * The only bridge into the renderer: read the config once at startup and
 * write it back on request. No Node, no fs, no remote -- the pet page is an
 * ordinary web page that happens to live in a transparent window.
 *
 * The read is synchronous on purpose. The first thing the pet page does is
 * fetch the model list, and that fetch has to carry the token; an async
 * round trip would render a frame first and 401 on the initial request.
 */
contextBridge.exposeInMainWorld("petConfig", {
  initial: ipcRenderer.sendSync("pet:config:get"),
  set: (patch) => ipcRenderer.invoke("pet:config:set", patch),
});