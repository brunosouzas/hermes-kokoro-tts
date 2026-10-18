"""hermes-kokoro-tts: local Kokoro text-to-speech for Hermes Agent on Apple Silicon."""

from __future__ import annotations


def register(ctx) -> None:
    from .kokoro_cli import make_handler, setup_parser
    from .kokoro_provider import KokoroTTSProvider

    def setting(key, default=None):
        value = ctx.get_config(key, default=default)
        return default if value in (None, "") else value

    ctx.register_tts_provider(KokoroTTSProvider(setting))
    ctx.register_cli_command(
        name="kokoro",
        help="Manage the local Kokoro text-to-speech server",
        setup_fn=setup_parser,
        handler_fn=make_handler(setting, lambda: KokoroTTSProvider(setting), save=getattr(ctx, "set_config", None)),
        description="Install, check and test the local Kokoro server used by tts.provider: kokoro-mlx.",
    )
