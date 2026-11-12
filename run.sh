#!/bin/bash
# Starts Jarvis on 127.0.0.1:8765. Phone access goes through `tailscale serve`.
cd "$(dirname "$0")" && exec .venv/bin/python server.py
