# SPDX-License-Identifier: Apache-2.0
"""granite-switch-systemone: a jev system-one API (`POST /v1/systemone`) bridge for
granite-switch's vLLM backend, mounted on the SAME server that already serves
`/v1/chat/completions`: one engine, one KV cache, one container.

See `plugin.py`'s module docstring for the full design. `renderer.py` is the one
adapter-specific piece this package carries today (the decision adapter's trained
prompt convention); `plugin.py` itself has no knowledge of any specific adapter.
"""

__version__ = "0.1.3"
