"""Shared appearance for the provided HW7 browser visualizers."""

VIEWER_STYLE = """
<style>
.info, .control-panel { display: none !important; }
.sidenav {
  width: min(290px, calc(100vw - 32px)); min-width: 0; height: auto;
  top: 16px; left: 16px; padding: 14px; border-radius: 12px;
  background: rgba(30,30,30,0.9); color: white;
  max-height: calc(100vh - 32px); overflow-y: auto;
}
.sidenav .label-div:first-child { display: none; }
.sidenav .label-div p { width: 100% !important; text-align: left;
  font: 13px/1.5 ui-monospace, monospace; color: white; }
.sidenav .button-button { width: 100%; cursor: pointer; }
</style>
"""
