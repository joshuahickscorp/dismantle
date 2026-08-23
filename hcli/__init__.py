"""HCLI product package — command surface / UI / status rendering.

Canonical import name: ``hcli``. ``tools.haider.hcli`` is gone.
Ownership packages: ``hcli.agentos``, ``hcli.genomes``,
``hcli.doctor``, ``hcli.gravity``, ``hcli.vmcp``. Runtime lives
as ``hcli.runtime`` / ``hcli.engine`` / ``hcli.backends``.
"""
from .cli import parse_haider_args, main
from .workspace import Workspace
from .controller import Controller
from .events import Event, EventBus

__all__ = ["parse_haider_args", "main", "Workspace", "Controller", "Event", "EventBus"]
