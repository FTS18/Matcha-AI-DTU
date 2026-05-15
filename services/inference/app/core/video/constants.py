from pathlib import Path

# -- Logo path (top-right watermark) ------------------------------------------
LOGO_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "apps"
    / "web"
    / "public"
    / "favicons"
    / "logo.png"
)

# -- Event-specific overlay configs -------------------------------------------
EVENT_CONFIG = {
    "GOAL": {"color": (0, 255, 100), "title": "GOAL", "transition": "circleopen"},
    "SAVE": {"color": (50, 180, 255), "title": "GREAT SAVE", "transition": "fadeblack"},
    "TACKLE": {"color": (255, 165, 0), "title": "TACKLE", "transition": "slideleft"},
    "FOUL": {"color": (255, 50, 50), "title": "FOUL", "transition": "fadeblack"},
    "CELEBRATION": {
        "color": (255, 215, 0),
        "title": "CELEBRATION",
        "transition": "circleopenclose",
    },
    "HIGHLIGHT": {
        "color": (200, 200, 255),
        "title": "KEY MOMENT",
        "transition": "fade",
    },
}

VALID_TRANSITIONS = {
    "fade",
    "fadeblack",
    "fadewhite",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "circlecrop",
    "rectcrop",
    "distance",
    "pixelize",
    "diagtl",
    "diagtr",
    "diagbl",
    "diagbr",
    "hlslice",
    "hrslice",
    "vuslice",
    "vdslice",
    "hblur",
    "fadegrays",
    "wipel",
    "wiper",
    "wipet",
    "wipeb",
}
