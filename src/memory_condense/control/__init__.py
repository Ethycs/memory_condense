"""Control plane: the status document, snapshots, bulk loading, and the local
control API. See ``docs/01 - Design/03 - System Controls.md``.

One writer (the proxy) publishes ``status.json``; every user surface renders
it or calls the control routes. Submodules are built independently against
that contract and must not import each other's internals beyond ``status``.
"""
