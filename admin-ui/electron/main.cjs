/**
 * Desktop pet shell.
 *
 * A transparent, always-on-top, click-through window showing one route of
 * the admin bundle -- the canvas and an input box, nothing else. The
 * rendering code is shared with the console; only the window chrome
 * differs.
 *
 * The server address lives in the userData config so the pet survives a
 * restart and can be pointed at another deployment without a rebuild.
 */
const { app, BrowserWindow, Tray, Menu, dialog, shell, nativeImage, ipcMain } = require("electron");
const path = require("node:path");
const fs = require("node:fs");

const WIDTH = 280;
const HEIGHT = 460;

let win = null;
let tray = null;

const CONFIG_PATH = () => path.join(app.getPath("userData"), "pet.json");

const DEFAULTS = { serverUrl: "http://127.0.0.1:8000", token: "" };

function readConfig() {
  try {
    return { ...DEFAULTS, ...JSON.parse(fs.readFileSync(CONFIG_PATH(), "utf8")) };
  } catch {
    // A missing or malformed config is not fatal -- the pet still starts
    // and the operator can repair the file from the tray.
    return { ...DEFAULTS };
  }
}

function writeConfig(patch) {
  const next = { ...readConfig(), ...patch };
  fs.mkdirSync(path.dirname(CONFIG_PATH()), { recursive: true });
  fs.writeFileSync(CONFIG_PATH(), JSON.stringify(next, null, 2));
  return next;
}

function createWindow() {
  win = new BrowserWindow({
    width: WIDTH,
    height: HEIGHT,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    // The window owns no shadow; Windows draws one for normal windows and
    // it renders as a grey halo around a transparent pet.
    hasShadow: false,
    resizable: false,
    skipTaskbar: true,
    // "#00000000" -- anything opaque here shows as a black box behind the
    // canvas on Windows.
    backgroundColor: "#00000000",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      // THE line that makes a desktop pet work at all. Electron pauses
      // requestAnimationFrame for unfocused or occluded windows, and a pet
      // is unfocused by definition -- it is a thing sitting on the
      // background. Without this it stands completely still.
      backgroundThrottling: false,
    },
  });

  // file:// has no server to resolve a path against, so the pet lives at a
  // hash route rather than a real one.
  win.loadFile(path.join(__dirname, "..", "dist", "index.html"), { hash: "pet" });
  win.setAlwaysOnTop(true, "screen-saver");

  // Click-through so the pet does not swallow clicks meant for whatever is
  // behind it. `forward: true` keeps mousemove flowing so the window can
  // notice the cursor arriving and turn interaction back on.
  win.setIgnoreMouseEvents(true, { forward: true });
  win.on("mouseleave", () => {
    if (win) win.setIgnoreMouseEvents(true, { forward: true });
  });
  win.on("closed", () => {
    win = null;
  });
}

function buildTray() {
  const iconPath = path.join(__dirname, "tray.png");
  // A missing icon must not stop startup; an empty NativeImage is valid
  // and simply renders as nothing.
  tray = new Tray(fs.existsSync(iconPath) ? iconPath : nativeImage.createEmpty());
  tray.setToolTip("Live2D 数字人");
  const refresh = () => {
    const cfg = readConfig();
    tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: "显示宠物", click: () => win?.show() },
        { label: "隐藏宠物", click: () => win?.hide() },
        { type: "separator" },
        // Shown, not editable: Electron has no built-in text-input dialog,
        // and a two-step message box that still cannot accept typing would
        // be worse than an honest label. Editing the file is the path.
        { label: `服务：${cfg.serverUrl}`, enabled: false },
        { label: "打开配置文件…", click: () => shell.openPath(CONFIG_PATH()) },
        {
          label: "查看服务端状态",
          click: async () => {
            try {
              const r = await fetch(`${cfg.serverUrl.replace(/\/$/, "")}/api/admin/live2d/health`);
              const body = await r.json();
              dialog.showMessageBox(win, {
                type: "info",
                title: "服务端状态",
                message: `SDK：${body.core_present ? "已就绪" : "缺失"}`,
                detail: `目录：${body.sdk_dir}`,
              });
            } catch (e) {
              dialog.showErrorBox("无法连接", `${cfg.serverUrl}\n${e.message}`);
            }
          },
        },
        { type: "separator" },
        { label: "退出", click: () => app.quit() },
      ]),
    );
  };
  refresh();
}

// The renderer reads its config synchronously at startup so the first fetch
// already carries the token -- an async bridge would render one frame
// without it and 401 on the very first model request.
ipcMain.on("pet:config:get", (e) => {
  e.returnValue = readConfig();
});
ipcMain.handle("pet:config:set", (_e, patch) => writeConfig(patch ?? {}));

// The renderer reports cursor movement (forwarded events reach it even while
// the window ignores input) and the main process lifts click-through.
ipcMain.on("pet:cursor", () => {
  if (win && !win.isDestroyed()) win.setIgnoreMouseEvents(false);
});

app.whenReady().then(() => {
  createWindow();
  buildTray();
});

app.on("window-all-closed", () => app.quit());