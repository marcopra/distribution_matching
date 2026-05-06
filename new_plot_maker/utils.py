"""Small console logger used by the plotting utilities."""

from __future__ import annotations


class Logger:
    COLORS = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }

    @classmethod
    def _color(cls, text: str, color: str | None = None) -> str:
        if color is None or color not in cls.COLORS:
            return text
        return f"{cls.COLORS[color]}{text}{cls.COLORS['reset']}"

    @classmethod
    def section(cls, title: str) -> None:
        print("")
        print(cls._color(f"== {title} ==", "bold"))

    @classmethod
    def subsection(cls, title: str) -> None:
        print("")
        print(cls._color(f"-- {title} --", "blue"))

    @classmethod
    def item(cls, text: str, color: str | None = None) -> None:
        print(f"* {cls._color(text, color)}")

    @staticmethod
    def detail(text: str, color: str | None = None) -> None:
        print(Logger._color(f"  {text}", color))
