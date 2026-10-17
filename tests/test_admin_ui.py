import pathlib
import re
import shutil
import subprocess

import pytest

DIST = pathlib.Path(__file__).resolve().parent.parent / "admin-ui" / "dist"
INDEX = DIST / "index.html"
_MINIMAL = '<!doctype html><html><body><div id="root"></div></body></html>'


@pytest.fixture()
def spa_dist():
    """Guarantee a built SPA index exists so create_app() registers the /admin mount.

    The real ``npm run build`` output is used when present; otherwise a minimal
    index.html is synthesized (and cleaned up) so the mount wiring is still
    exercised on a source-only checkout.
    """
    created = not INDEX.exists()
    if created:
        DIST.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(_MINIMAL, encoding="utf-8")
    yield INDEX
    if created:
        INDEX.unlink(missing_ok=True)
        try:
            DIST.rmdir()
        except OSError:
            pass


@pytest.fixture()
def ui_client(spa_dist, client):
    # spa_dist is set up before client, so create_app() sees the dist directory.
    return client


def test_admin_mount_serves_spa_index(ui_client):
    resp = ui_client.get("/admin/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="root"' in resp.text


def test_admin_root_redirects_to_index(ui_client):
    # Assert the redirect itself, not just the followed 200: a regression that
    # stops /admin redirecting to /admin/ must fail this test.
    resp = ui_client.get("/admin", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)
    assert resp.headers["location"].endswith("/admin/")


def test_admin_deep_link_serves_spa_index(ui_client):
    # A client-side route (no file extension) must fall back to the SPA index
    # so a refresh / deep link under the /admin mount does not 404.
    resp = ui_client.get("/admin/avatars")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="root"' in resp.text


def test_admin_missing_asset_still_404s(ui_client):
    # A request for a real file (has an extension) must NOT fall back to index.
    resp = ui_client.get("/admin/assets/does-not-exist.js")
    assert resp.status_code == 404


def test_browser_router_uses_admin_basename():
    # The SPA is served under /admin, so client URLs must be scoped to that
    # mount; otherwise refresh/deep-links escape it. Not observable from the
    # server, so guard the source directly.
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui"
        / "src"
        / "main.tsx"
    ).read_text(encoding="utf-8")
    assert 'basename="/admin"' in source


def _avatars_source():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui"
        / "src"
        / "pages"
        / "Avatars.tsx"
    ).read_text(encoding="utf-8")


def test_avatar_upload_accepts_video():
    # The upload control must offer videos as well as images; the server enforces
    # the real limits, but the widget has to let a video be chosen at all.
    assert 'accept="image/*,video/*"' in _avatars_source()


def test_avatar_create_has_synchronous_reentry_guard():
    # The create flow must run inside the guard's locked section, with the
    # button state always cleared; the behavioral test below proves the lock
    # actually covers the probe await.
    source = _avatars_source()
    assert "createReentryGuard()" in source
    assert "creatingGuard.run(async () =>" in source
    assert "setCreating(true);" in source
    assert "      } finally {\n        setCreating(false);\n      }" in source


def test_reentry_guard_blocks_double_submit_behaviorally():
    # Executes the real guard module with node:test (Node strips TS types):
    # two calls while the first task is awaiting its probe must run exactly
    # one task. Source-text assertions cannot detect guard placement; this
    # test can.
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the behavioral re-entry guard test")
    ui_root = pathlib.Path(__file__).resolve().parent.parent / "admin-ui"
    result = subprocess.run(
        [node, "--test", "src/pages/reentryGuard.test.ts"],
        cwd=ui_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_avatar_upload_hint_asks_for_closed_mouth():
    assert "视频中请保持闭嘴、不说话" in _avatars_source()


def test_avatar_create_form_exposes_idle_overrides():
    # The create form surfaces the two idle-video knobs (idle_crf + idle_preset)
    # so the operator can pick per-avatar quality at upload time. Both must
    # be appended to the FormData so the server stores them on the avatar
    # row. No other tunables (the dropped face-crop / quality-threshold ones)
    # should be editable here.
    source = _avatars_source()
    assert 'body.append("idle_crf"' in source, (
        "create form must submit idle_crf so the server can persist the override"
    )
    assert 'body.append("idle_preset"' in source, (
        "create form must submit idle_preset so the server can persist the override"
    )
    # No tunables that the user said to drop should appear as form fields.
    assert 'body.append("face_crop_size"' not in source
    assert 'body.append("bbox_shift"' not in source
    assert 'body.append("min_sharpness"' not in source


def test_avatar_create_form_has_no_settings_link():
    # The Settings tab is gone. The upload card must not link to it, and the
    # read-only "高级参数" summary that pointed operators there is removed.
    source = _avatars_source()
    assert 'navigate("/settings")' not in source
    assert "调整参数（设置 → 制作）" not in source


def test_avatar_upload_drops_animate_toggle():
    # The "先把照片动起来" toggle has moved off the avatar upload form onto
    # its own "/activate-portrait" tab. Lock that removal so a future
    # "let me put it back here" cannot quietly undo the move.
    source = _avatars_source()
    assert "先把照片动起来" not in source, (
        "animate checkbox must live on /activate-portrait, not on the avatar upload card"
    )
    assert 'name="animate"' not in source, "Form.Item must not bind to animate anymore"
    assert 'body.append("animate"' not in source, (
        "the avatar upload form data must not include the animate key"
    )


def _activate_portrait_source():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "pages" / "ActivatePortrait.tsx"
    ).read_text(encoding="utf-8")


def test_activate_portrait_exposes_param_inputs():
    # The "形象活化" tab must surface prompt / num_frames / steps inputs so
    # the operator can override them per generation. Each is sent to
    # /api/admin/i2v as a form field; the backend then validates per its own
    # animator's contract.
    source = _activate_portrait_source()
    for key in ("num_frames", "steps", "prompt"):
        assert f'append("{key}"' in source or f"append('{key}'" in source, (
            f"ActivatePortrait must append '{key}' to its FormData"
        )


def test_activate_portrait_routes_to_i2v_with_backend():
    # The submit button POSTs to /api/admin/i2v with the picked backend name
    # so the server can dispatch to the correct animator. Submission is
    # async: the endpoint returns 202 + task_id, and the page follows up by
    # polling /api/admin/tasks/{id} for progress.
    source = _activate_portrait_source()
    assert '"/api/admin/i2v"' in source, "ActivatePortrait must POST to /api/admin/i2v"
    assert 'body.append("backend"' in source, (
        "the form must include the picked backend so the dispatcher can route"
    )
    # Polling endpoint is wired so the page can show progress while the
    # encoder runs. The page must NOT block on the i2v response anymore.
    assert "/api/admin/tasks/" in source, (
        "ActivatePortrait must poll /api/admin/tasks/{id} to track async i2v progress"
    )


def test_activate_portrait_offers_use_video_to_create_avatar():
    # Once the i2v task is done, the operator must be able to take the
    # generated mp4 straight to the avatar create form (形象管理 -> 新建形象).
    # The page fetches the result bytes from /api/admin/tasks/{id}/result.mp4
    # and navigates with the file pre-loaded.
    source = _activate_portrait_source()
    assert "/result.mp4" in source, (
        "ActivatePortrait must read the rendered mp4 from the task result endpoint"
    )
    assert "用此视频创建形象" in source, (
        "After a successful render, the page must show a 'use this video to create an avatar' button"
    )
    assert 'navigate("/avatars"' in source, (
        "The button must navigate to /avatars so the create form can pick up the preloaded file"
    )


def test_activate_portrait_has_records_table_for_i2v_task():
    # The 图生视频 page surfaces a list of past generations so the operator
    # can view / delete / download / use-as-avatar any of them, mirroring the
    # 形象管理 table pattern. The list is fed from /api/admin/tasks.
    source = _activate_portrait_source()
    assert "/api/admin/tasks" in source, (
        "ActivatePortrait must GET /api/admin/tasks to populate the records list"
    )
    # Antd Table is the list primitive (not a bare <ul>).
    assert "Table" in source, (
        "ActivatePortrait must render an Antd Table for the records list"
    )


def test_activate_portrait_drawer_exposes_view_delete_download_and_use_as_avatar():
    # The 查看 action opens a Drawer with the operations an operator needs
    # on an existing record: preview, download, use-as-avatar, delete.
    # Lock the action set so a future shrink ("we don't need download
    # anymore") cannot silently drop a button the operator relies on.
    source = _activate_portrait_source()
    assert "Drawer" in source, (
        "ActivatePortrait must render a Drawer for the 查看 view (not a Modal/page)"
    )
    # Download: the byte fetch from the result endpoint must still be wired.
    assert "/result.mp4" in source, (
        "The drawer's 下载 button must read bytes from /api/admin/tasks/{id}/result.mp4"
    )
    # Use-as-avatar: the action still ships the file over to /avatars.
    assert "用此视频创建形象" in source, (
        "The drawer must keep the 'use this video to create an avatar' shortcut"
    )
    # Delete: the drawer (or the row) must issue DELETE /api/admin/tasks/{id}.
    assert '"DELETE"' in source, (
        "ActivatePortrait must call DELETE on /api/admin/tasks/{id} for the delete action"
    )
    assert "Popconfirm" in source, (
        "The delete action must confirm before issuing DELETE"
    )


def test_activate_portrait_drops_right_pane_preview_card():
    # The old design rendered the current result in a right-side Card next
    # to the form. That surface was redundant once the records list +
    # 查看 drawer cover the same need; dropping it brings the page in line
    # with 形象管理's stack (form-on-top, list-below) layout. The
    # identifying structure was the <Row gutter> + two <Col xs={24} md={12}>
    # columns; the form card on the left alone does not need them.
    source = _activate_portrait_source()
    assert '<Card title="生成结果">' not in source, (
        "The right-pane '生成结果' Card must be removed; the record drawer covers it"
    )
    assert "本次生成实际使用值" not in source, (
        "The right-pane params summary was duplicated info; the drawer surfaces params from task.params instead"
    )
    assert "Col xs={24} md={12}" not in source, (
        "Two-column <Row gutter> layout must be gone; the form now stacks above the records list"
    )


def test_activate_portrait_inserts_submitted_task_into_records_list():
    # When the operator clicks 生成视频 and the server returns a task_id,
    # the new row must appear in the records table -- the user explicitly
    # asked for "点生成视频后，记录里展示进度和状态". The submission path
    # already exists; lock the list-mutation side too.
    source = _activate_portrait_source()
    has_setter = (
        "setItems" in source
        or "setRecords" in source
        or "setTasks" in source
    )
    assert has_setter, (
        "After submit, ActivatePortrait must update the records list so the new row shows progress + status"
    )


def test_app_nav_renames_i2v_tab_to_image_to_video():
    # The '形象活化' label was too abstract; it now reads 图生视频. Lock the
    # rename so a future revert is caught. The URL stays at /activate-portrait
    # to avoid gratuitous churn.
    app = _app_source()
    assert "图生视频" in app, "App.tsx must show 图生视频 as the i2v tab label"
    assert "形象活化" not in app, "Old abstract label must be gone"


def test_app_nav_drops_tasks_tab():
    # The 任务 tab has been merged into the 图生视频 page (records list lives
    # under /activate-portrait). Lock the removal so a future revert cannot
    # silently split the surface in two.
    app = _app_source()
    assert "/tasks" not in app, "App.tsx must not register a /tasks route"
    assert "import Tasks" not in app, "App.tsx must not import the Tasks page"
    assert "任务" not in app, "App.tsx nav must not show 任务 anymore"


def test_tasks_page_file_is_removed():
    # The page file must be deleted (so TS doesn't keep an orphan surface
    # around) along with the route. An orphaned page file would still
    # compile, but a stale import in App.tsx would 404 the operator.
    pages_dir = (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "pages"
    )
    assert not (pages_dir / "Tasks.tsx").exists(), "Tasks.tsx must be deleted"


def test_avatars_page_accepts_prefilled_video_from_navigation():
    # When the operator clicks 'use this video' on the 图生视频 page, the
    # avatar create form must seed its upload list from the navigation
    # state. Lock the field name so a rename doesn't silently drop the file.
    source = _avatars_source()
    assert "prefilledFile" in source, (
        "Avatars.tsx must read location.state.prefilledFile to seed the upload"
    )


def test_avatars_declares_prefilled_ref_before_use_state():
    # Regression guard for a Temporal Dead Zone crash: ``initialPrefilledRef``
    # is read inside a ``useState(() => {...})`` initializer, which React
    # invokes on first render -- BEFORE the line of code that declares a
    # ``const`` would execute. Reading the const then throws
    # ReferenceError. Lock the declaration order so a future re-order
    # can't silently reintroduce the crash.
    source = _avatars_source()
    decl_idx = source.find("const initialPrefilledRef")
    assert decl_idx >= 0, "Avatars.tsx must declare initialPrefilledRef"
    # The first read inside the file must come AFTER the declaration.
    first_read = source.find("initialPrefilledRef.current")
    assert first_read >= 0, "Avatars.tsx must read initialPrefilledRef somewhere"
    assert decl_idx < first_read, (
        f"initialPrefilledRef declared at offset {decl_idx} but first used at offset {first_read}; "
        "a useState(() => ...) initializer runs on first render and reads the const before its declaration line executes (TDZ)"
    )


def test_avatars_wraps_prefilled_file_without_assign():
    # Regression guard for two consecutive browser-side crashes on /avatars
    # after the i2v handoff:
    #
    # 1) ``Object.assign(seeded, { ..., lastModifiedDate: ... })`` threw
    #    ``TypeError: Cannot set property lastModifiedDate of #<File>``
    #    because ``File.prototype.lastModifiedDate`` is a legacy read-only
    #    getter on modern browsers.
    # 2) A first attempted fix wrapped the File via ``Object.create`` and
    #    used ``Object.defineProperty`` to add the extra fields. That
    #    satisfied RcFile's shape but stripped the ``[[Blob]]`` internal
    #    slot, so ``URL.createObjectURL``/``FileReader``/``File.slice``
    #    later threw ``TypeError: Illegal invocation`` when AntD's preview
    #    pipeline and our own ``videoSeconds`` helper reached for them.
    #
    # The correct fix uses the original File as-is (so the [[Blob]] slot
    # is intact) and only writes ``uid`` -- a plain string property whose
    # assignment is unconstrained. ``lastModifiedDate`` is intentionally
    # NEVER set; AntD v5's runtime never reads it (rc-upload v4 only
    # touches ``uid``/``name``/``type``/``size``/``lastModified``).
    source = _avatars_source()
    assert "Object.assign(seeded" not in source, (
        "Avatars.tsx must not Object.assign onto a File: "
        "File.prototype.lastModifiedDate is a read-only getter in modern browsers."
    )
    assert "Object.create(seeded)" not in source, (
        "Avatars.tsx must not wrap the prefilled File via Object.create(seeded); "
        "an Object.create proxy inherits the prototype but NOT the [[Blob]] "
        "internal slot, so URL.createObjectURL / FormData.append throw "
        "'TypeError: Illegal invocation' on the wrapper."
    )
    # We deliberately never write to ``.lastModifiedDate``: it's a read-only
    # legacy getter on modern browsers, and AntD v5's runtime (rc-upload v4)
    # doesn't read it. The exact write patterns we forbid are: assignment via
    # ``Object.assign`` (above) or ``Object.create`` wrapper (above); the
    # current code path uses the seed File as-is and only writes ``uid``.
    # Match the two *write-attempt* shapes, not the bare identifier -- the
    # identifier appears in this test's own comments to document why it's
    # avoided.
    assert not _writes_last_modified_date(source), (
        "Avatars.tsx must not assign to .lastModifiedDate on a File: the field is "
        "a read-only legacy getter in modern browsers and AntD v5's runtime never "
        "reads it."
    )
    assert "rcFile.uid" in source or "originFileObj.uid" in source, (
        "Avatars.tsx must set rcFile.uid to satisfy AntD's RcFile shape (File & "
        "{ uid: string }); without it the upload list can't track the entry."
    )


def _writes_last_modified_date(source: str) -> bool:
    # Detect the two write patterns we care about: anything that looks like an
    # assignment / property definition that puts a value into ``.lastModifiedDate``.
    # We look at lines *outside* of comments by stripping them first, so this
    # test stays robust against text that merely *mentions* the field.
    no_comments = []
    for line in source.splitlines():
        stripped = line.split("//", 1)[0]
        no_comments.append(stripped)
    code = "\n".join(no_comments)
    # Match ``lastModifiedDate: <something>`` (property def) or
    # ``.lastModifiedDate = <something>`` (plain assignment).
    return bool(
        re.search(r"\blastModifiedDate\s*:", code)
        or re.search(r"\.lastModifiedDate\s*=", code)
    )


def test_activate_portrait_greys_out_backends_that_dont_fit_current_gpu():
    # The server's meta() carries ``fits_current_gpu``; the page must
    # render those options as ``disabled`` so the operator doesn't click
    # + wait + see a VRAM error. Variants without a verdict (``null``)
    # stay clickable -- the dispatcher still refuses if the host really
    # can't run them.
    source = _activate_portrait_source()
    # The dropdown's option builder reads ``fits_current_gpu`` and emits
    # ``disabled`` for false, ``true``/``null`` otherwise.
    assert "fits_current_gpu" in source, (
        "ActivatePortrait must read fits_current_gpu from /api/admin/backends"
    )
    assert "disabled: unavailable" in source, (
        "the Select option must be marked disabled when fits_current_gpu === false"
    )


def test_activate_portrait_is_first_tab():
    # The user-facing requirement is that "形象活化" is the first menu entry;
    # a regression that re-sorts the nav back to "形象管理" first must fail.
    app_source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "App.tsx"
    ).read_text(encoding="utf-8")
    # The /activate-portrait key comes first among the menu entries.
    activate_idx = app_source.find('"/activate-portrait"')
    avatars_idx = app_source.find('"/avatars"')
    assert 0 <= activate_idx < avatars_idx, (
        "形象活化 (key /activate-portrait) must be the first nav entry, before /avatars"
    )
    # The Routes block also lists /activate-portrait before /avatars.
    routes_idx = app_source.find('path="/activate-portrait"')
    avatars_route_idx = app_source.find('path="/avatars"')
    assert 0 <= routes_idx < avatars_route_idx, (
        "/activate-portrait route must be declared before /avatars"
    )


def test_nav_label_is_图生视频_not_形象活化():
    # The tab was renamed from 形象活化 (too abstract) to 图生视频 (concrete:
    # describes what the button does). Lock both directions so a future revert
    # of either label is caught.
    app_source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "App.tsx"
    ).read_text(encoding="utf-8")
    assert "图生视频" in app_source, (
        "the page must be labelled 图生视频"
    )
    assert "形象活化" not in app_source, (
        "the old abstract label must not appear anywhere in the nav"
    )


def _app_source():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "App.tsx"
    ).read_text(encoding="utf-8")


def _types_source():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "types.ts"
    ).read_text(encoding="utf-8")


def test_settings_page_is_gone():
    # The Settings tab was removed: the page file must be deleted (so TS
    # doesn't keep an orphan surface around) and App.tsx must not import or
    # route it. The nav entry must also be gone so the tab doesn't render
    # as an empty row.
    pages_dir = (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "pages"
    )
    assert not (pages_dir / "Settings.tsx").exists(), "Settings.tsx must be removed"
    app = _app_source()
    assert 'pages/Settings"' not in app, "App.tsx must not import Settings"
    assert 'path="/settings"' not in app, "App.tsx must not route /settings"
    assert '"/settings"' not in app, "App.tsx must not have a /settings nav entry"


def test_types_drops_non_idle_tunables():
    # The TunableParams surface is what the create form talks to. The user-
    # visible knobs on the create form are idle_crf + idle_preset; everything
    # else has moved off the UI. The old ten-knob surface stays in the
    # backend's GET /api/admin/settings for audit-row completeness, but the
    # TypeScript shape that drives the form should reflect only what the
    # form actually edits. Drop the others from the type.
    source = _types_source()
    # Lock the dropped keys out of the UI-side TunableParams surface.
    assert "face_crop_size" not in source, "face_crop_size is inert -- drop from UI surface"
    assert "bbox_shift" not in source, "bbox_shift is inert -- drop from UI surface"
    assert "min_sharpness" not in source
    assert "min_occupancy" not in source
    assert "min_face_width_px" not in source
    assert "min_nose_ratio" not in source
    assert "video_min_frames" not in source
    assert "min_detection_confidence" not in source
    assert "default_liveportrait_driver" not in source
    # The two idle knobs the create form still edits must remain.
    assert "idle_crf" in source
    assert "idle_preset" in source


def _detail_source():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "pages" / "AvatarDetail.tsx"
    ).read_text(encoding="utf-8")


def test_detail_preview_is_a_loop_video_with_image_fallback():
    source = _detail_source()
    assert "idle.mp4" in source
    assert "<video" in source or "video" in source
    assert "muted" in source and "autoPlay" in source and "loop" in source
    assert "normalized.png" in source
    assert "preview.png" not in source


def test_detail_has_no_drive_level_slider():
    source = _detail_source()
    assert "驱动电平" not in source
