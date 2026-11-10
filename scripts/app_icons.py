"""Render the app icon files from app/static/brand/app-icon.svg; run again after changing it.

    uv run python -m scripts.app_icons      (macOS: uses AppKit to draw the SVG and iconutil)

Writes next to the SVG:
  app-icon.png   512 px: the Dock icon when running from source, and for Linux
  app-icon.icns  the macOS app bundle icon (PyInstaller)
  app-icon.ico   the Windows executable and taskbar icon (PyInstaller)
"""

import subprocess
import tempfile
from pathlib import Path

import AppKit
from PIL import Image

from app import config

BRAND = config.BRAND_DIR
SOURCE = BRAND / "app-icon.svg"


def render(svg: AppKit.NSImage, px: int, out: Path) -> None:
    """Draw the SVG into a px × px PNG."""
    rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, px, px, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0
    )
    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    svg.drawInRect_(AppKit.NSMakeRect(0, 0, px, px))
    AppKit.NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, None).writeToFile_atomically_(str(out), True)


def main() -> None:
    svg = AppKit.NSImage.alloc().initWithContentsOfFile_(str(SOURCE))
    if svg is None:
        raise SystemExit(f"Can't read {SOURCE}")

    render(svg, 512, BRAND / "app-icon.png")  # the Dock shows at most 512 px

    # macOS: an .iconset folder of the standard sizes, packed by iconutil.
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "app-icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            render(svg, size, iconset / f"icon_{size}x{size}.png")
            render(svg, size * 2, iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(BRAND / "app-icon.icns")], check=True)

    # Windows: one .ico holding the sizes Explorer and the taskbar use.
    with Image.open(BRAND / "app-icon.png") as png:
        png.save(BRAND / "app-icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

    print(f"Wrote app-icon.png, .icns and .ico in {BRAND}")


if __name__ == "__main__":
    main()
