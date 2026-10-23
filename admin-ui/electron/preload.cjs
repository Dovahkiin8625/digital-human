const { contextBridge, ipcRenderer } = require("electron");

/**
 * The only bridge into the renderer: read the config once at startup, write
 * it back on request, and report cursor movement. No Node, no fs, no remote
 * -- the pet page is an ordinary web page that happens to live in a
 * transparent window.
 *
 * The config read is synchronous on purpose. The first thing the pet page
 * does is fetch the model list, and that fetch has to carry the token; an
 * async round trip would render a frame first and 401 the initial request.
 */
contextBridge.exposeInMainWorld("petConfig", {
  initial: ipcRenderer.sendSync("pet:config:get"),
  set: (patch) => ipcRenderer.invoke("pet:config:set", patch),
  // Click-through detection MUST come from the renderer. The main-process
  // `win.on("mousemove")` does not fire for an unfocused window, and a pet
  // is unfocused by definition -- so listening there means the window can
  // never stop being click-through and its input box is unusable.
  // `setIgnoreMouseEvents(true, {forward: true})` forwards these moves here.
  reportCursor: () => ipcRenderer.send("pet:cursor"),
});
