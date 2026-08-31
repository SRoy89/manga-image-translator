import numpy as np
from manga_translator.rendering import resize_regions_to_font_size
from manga_translator.utils import TextBlock


def test_resize_regions_does_not_expand_target_polygon_for_long_vertical_text():
    img = np.full((500, 500, 3), 255, dtype=np.uint8)
    region = TextBlock(
        lines=[[[100, 100], [160, 100], [160, 420], [100, 420]]],
        texts=["きっと"],
        font_size=80,
        translation="一定和我一样呢",
        source_lang="ja",
        target_lang="CHS",
    )
    [dst_points] = resize_regions_to_font_size(
        img, [region], font_size_fixed=None, font_size_offset=0, font_size_minimum=8,
    )
    assert np.array_equal(dst_points, region.min_rect)


def test_long_translation_does_not_increase_font_size():
    img = np.full((500, 500, 3), 255, dtype=np.uint8)
    region = TextBlock(
        lines=[[[100, 100], [160, 100], [160, 420], [100, 420]]],
        texts=["きっと"],
        font_size=80,
        translation="一定和我一样呢",
        source_lang="ja",
        target_lang="CHS",
    )
    original = region.font_size
    resize_regions_to_font_size(
        img, [region], font_size_fixed=None, font_size_offset=0, font_size_minimum=8,
    )
    assert region.font_size <= original
