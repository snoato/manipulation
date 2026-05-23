"""Unit tests for the AssetSet abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from tampanda.scenes.assets.asset_set import (
    Asset,
    AssetSet,
    MultiTierShelf,
    Shelf,
    YCB_PROXY_ITEMS,
    make_generic_boxes,
    make_ycb_proxy,
)


class TestAsset:
    def test_minimal_construction(self):
        a = Asset(asset_id="b0", half_extents=(0.025, 0.025, 0.025))
        assert a.full_size == (0.05, 0.05, 0.05)

    def test_rejects_empty_id(self):
        with pytest.raises(ValueError, match="non-empty"):
            Asset(asset_id="", half_extents=(0.01, 0.01, 0.01))

    def test_rejects_double_underscore_in_id(self):
        with pytest.raises(ValueError, match="must not contain '__'"):
            Asset(asset_id="bad__name", half_extents=(0.01, 0.01, 0.01))

    def test_rejects_zero_extent(self):
        with pytest.raises(ValueError, match="half_extents must be positive"):
            Asset(asset_id="b", half_extents=(0.0, 0.01, 0.01))

    def test_renders_valid_xml(self):
        a = Asset(
            asset_id="meat",
            half_extents=(0.05, 0.029, 0.041),
            color=(0.8, 0.3, 0.3, 1.0),
        )
        xml = a.render_template_xml()
        assert "<body" in xml
        assert "<joint" in xml and 'type="free"' in xml
        assert 'type="box"' in xml
        assert 'size="0.05 0.029 0.041"' in xml
        assert 'rgba="0.8000 0.3000 0.3000 1.0000"' in xml
        # Important: the SceneBuilder relies on the _-prefix convention.
        assert 'name="_freejoint"' in xml
        assert 'name="_geom"' in xml

    def test_renders_xml_without_color_omits_rgba(self):
        a = Asset(asset_id="b", half_extents=(0.01, 0.01, 0.01))
        xml = a.render_template_xml()
        assert "rgba=" not in xml


class TestAssetSet:
    def test_iteration_preserves_order(self):
        a0 = Asset("z", (0.01, 0.01, 0.01))
        a1 = Asset("a", (0.02, 0.02, 0.02))
        s = AssetSet([a0, a1])
        assert s.ids() == ["z", "a"]
        assert list(s)[0] is a0

    def test_lookup_by_id(self):
        a = Asset("box", (0.01, 0.01, 0.01))
        s = AssetSet([a])
        assert s["box"] is a
        assert "box" in s
        assert "missing" not in s

    def test_duplicate_id_rejected(self):
        a = Asset("dup", (0.01, 0.01, 0.01))
        b = Asset("dup", (0.02, 0.02, 0.02))
        with pytest.raises(ValueError, match="duplicate asset_id"):
            AssetSet([a, b])

    def test_filter_by_prefix(self):
        s = AssetSet([
            Asset("blocker_0", (0.01, 0.01, 0.01)),
            Asset("blocker_1", (0.01, 0.01, 0.01)),
            Asset("target", (0.02, 0.02, 0.02)),
        ])
        sub = s.filter("blocker")
        assert sub.ids() == ["blocker_0", "blocker_1"]

    def test_materialised_writes_one_file_per_asset(self, tmp_path: Path):
        s = AssetSet([
            Asset("a", (0.01, 0.01, 0.01)),
            Asset("b", (0.02, 0.02, 0.02)),
        ])
        with s.materialised() as paths:
            assert set(paths) == {"a", "b"}
            for asset_id, p in paths.items():
                assert p.exists()
                content = p.read_text()
                assert "<body" in content
                assert content == s[asset_id].render_template_xml()
            tempdir = paths["a"].parent
            assert tempdir.exists()
        # Tempdir is cleaned up after the with block exits
        assert not tempdir.exists()


class TestGenericBoxes:
    def test_count_matches_sizes(self):
        s = make_generic_boxes("blocker", sizes=[(0.025, 0.025, 0.025)] * 5)
        assert len(s) == 5
        assert s.ids() == [f"blocker_{i}" for i in range(5)]

    def test_color_palette_cycles(self):
        # 10 boxes, 8 default colors → colors[0] == colors[8]
        s = make_generic_boxes("b", sizes=[(0.025, 0.025, 0.025)] * 10)
        assets = list(s)
        assert assets[0].color == assets[8].color

    def test_custom_colors_respected(self):
        red = (1.0, 0.0, 0.0, 1.0)
        s = make_generic_boxes(
            "b", sizes=[(0.025, 0.025, 0.025)] * 2, colors=[red, red],
        )
        for a in s:
            assert a.color == red


class TestShelf:
    def test_minimal_construction(self):
        s = Shelf(asset_id="shelf", interior_size=(0.3, 0.4, 0.2))
        assert s.exterior_size == (
            0.3 + 2 * s.wall_thickness,
            0.4 + 2 * s.wall_thickness,
            0.2 + 2 * s.wall_thickness,
        )

    def test_rejects_invalid_face(self):
        with pytest.raises(ValueError, match="open face"):
            Shelf(asset_id="s", interior_size=(0.1, 0.1, 0.1), open_faces=("up",))

    def test_rejects_zero_interior(self):
        with pytest.raises(ValueError, match="interior_size must be positive"):
            Shelf(asset_id="s", interior_size=(0.1, 0.0, 0.1))

    def test_rejects_duplicate_open_face(self):
        with pytest.raises(ValueError, match="duplicate face"):
            Shelf(asset_id="s", interior_size=(0.1, 0.1, 0.1), open_faces=("+x", "+x"))

    def test_rejects_all_faces_open(self):
        with pytest.raises(ValueError, match="at least one face must remain closed"):
            Shelf(asset_id="s", interior_size=(0.1, 0.1, 0.1),
                  open_faces=("+x", "-x", "+y", "-y", "+z", "-z"))

    def test_open_faces_must_be_tuple(self):
        with pytest.raises(TypeError, match="must be a tuple"):
            Shelf(asset_id="s", interior_size=(0.1, 0.1, 0.1), open_faces=["+x"])

    def test_default_omits_pos_x_wall(self):
        s = Shelf(asset_id="shelf", interior_size=(0.3, 0.4, 0.2))
        xml = s.render_template_xml()
        assert "_wall_pos_x" not in xml  # the (default) open face
        # the other walls are present
        assert "_wall_neg_x" in xml
        assert "_wall_pos_y" in xml
        assert "_wall_neg_y" in xml
        assert "_floor" in xml
        assert "_top" in xml

    def test_two_open_faces_makes_a_tunnel(self):
        # HAL access-style — open in +x and -x (front and back).  Side
        # panels and top/bottom decks remain.  Five wall geoms reduces to 4.
        s = Shelf(asset_id="shelf", interior_size=(0.3, 0.4, 0.2),
                  open_faces=("+x", "-x"))
        xml = s.render_template_xml()
        assert "_wall_pos_x" not in xml
        assert "_wall_neg_x" not in xml
        assert "_wall_pos_y" in xml
        assert "_wall_neg_y" in xml
        assert "_floor" in xml
        assert "_top" in xml
        assert xml.count("<geom") == 4

    def test_no_top_via_open_faces(self):
        s = Shelf(asset_id="shelf", interior_size=(0.3, 0.4, 0.2),
                  open_faces=("+x", "+z"))
        xml = s.render_template_xml()
        assert "_top" not in xml
        assert "_floor" in xml

    def test_xml_is_a_single_body_fragment(self):
        import xml.etree.ElementTree as ET
        s = Shelf(asset_id="shelf", interior_size=(0.3, 0.4, 0.2))
        xml = s.render_template_xml()
        wrapper = ET.fromstring(f"<_root>{xml}</_root>")
        bodies = [c for c in wrapper if c.tag == "body"]
        assert len(bodies) == 1

    def test_wall_count_matches_open_faces(self):
        # +x open: floor + top + 3 vertical walls = 5
        s = Shelf(asset_id="s", interior_size=(0.3, 0.4, 0.2), open_faces=("+x",))
        assert s.render_template_xml().count("<geom") == 5

        # +x and -x open (deck-style): 4
        s = Shelf(asset_id="s", interior_size=(0.3, 0.4, 0.2), open_faces=("+x", "-x"))
        assert s.render_template_xml().count("<geom") == 4

        # +x and +z open (open-front, open-top): 4
        s = Shelf(asset_id="s", interior_size=(0.3, 0.4, 0.2), open_faces=("+x", "+z"))
        assert s.render_template_xml().count("<geom") == 4


class TestMultiTierShelf:
    def test_minimal_construction(self):
        s = MultiTierShelf(
            asset_id="access_shelf",
            deck_size=(0.40, 0.50),
            deck_levels=(0.40, 0.62),
        )
        assert s.top_deck_top_z == pytest.approx(0.62)
        assert s.deck_top_z(0) == pytest.approx(0.40)

    def test_renders_two_decks_and_four_legs(self):
        s = MultiTierShelf(
            asset_id="access_shelf",
            deck_size=(0.40, 0.50),
            deck_levels=(0.40, 0.62),
        )
        xml = s.render_template_xml()
        # 4 legs + 2 decks = 6 geoms
        assert xml.count("<geom") == 6
        assert "_leg_0" in xml and "_leg_3" in xml
        assert "_deck_0" in xml and "_deck_1" in xml

    def test_renders_three_decks(self):
        s = MultiTierShelf(
            asset_id="three_tier",
            deck_size=(0.40, 0.50),
            deck_levels=(0.30, 0.50, 0.70),
        )
        assert s.render_template_xml().count("_deck_") == 3

    def test_xml_is_a_single_body_fragment(self):
        import xml.etree.ElementTree as ET
        s = MultiTierShelf(
            asset_id="t",
            deck_size=(0.40, 0.50),
            deck_levels=(0.40, 0.62),
        )
        xml = s.render_template_xml()
        wrapper = ET.fromstring(f"<_root>{xml}</_root>")
        bodies = [c for c in wrapper if c.tag == "body"]
        assert len(bodies) == 1

    def test_rejects_unsorted_deck_levels(self):
        with pytest.raises(ValueError, match="must be ascending"):
            MultiTierShelf(asset_id="t", deck_size=(0.4, 0.5),
                           deck_levels=(0.62, 0.40))

    def test_rejects_zero_deck_size(self):
        with pytest.raises(ValueError, match="deck_size must be positive"):
            MultiTierShelf(asset_id="t", deck_size=(0.0, 0.5))

    def test_rejects_empty_deck_levels(self):
        with pytest.raises(ValueError, match="deck_levels must be non-empty"):
            MultiTierShelf(asset_id="t", deck_size=(0.4, 0.5),
                           deck_levels=())


class TestAssetSetMixedContents:
    def test_can_hold_assets_and_shelves_together(self):
        s = AssetSet([
            Asset("box_0", (0.025, 0.025, 0.025)),
            Shelf("shelf", interior_size=(0.30, 0.40, 0.20)),
        ])
        assert s.ids() == ["box_0", "shelf"]
        # Both should materialise into XML files
        with s.materialised() as paths:
            assert "box_0" in paths and "shelf" in paths
            assert "type=\"box\"" in paths["shelf"].read_text()


class TestYcbProxy:
    def test_known_items(self):
        s = make_ycb_proxy(["meat_can", "mustard_bottle"])
        assert s.ids() == ["meat_can", "mustard_bottle"]

    def test_unknown_item_rejected(self):
        with pytest.raises(KeyError, match="unknown YCB proxy item"):
            make_ycb_proxy(["nonexistent_thing"])

    def test_dimensions_match_ycb_proxy_table(self):
        # Sizes are scaled down from real YCB so the Franka can grasp
        # each item (gripper max opening = 0.08 m; FRONT_X grasp axis
        # is world-x → ``half_x`` ≤ 0.035 m).
        s = make_ycb_proxy(["meat_can"])
        meat = s["meat_can"]
        assert meat.half_extents == (0.030, 0.029, 0.041)

    def test_all_items_fit_gripper_in_front_grasp(self):
        # Every YCB-proxy item must be FRONT_X-graspable: along the
        # finger spread axis (world-x), block width = 2 * half_x must
        # leave margin under the 0.08 m gripper opening.
        from tampanda.scenes.assets.asset_set import _YCB_PROXY_DIMENSIONS
        for name, half in _YCB_PROXY_DIMENSIONS.items():
            assert 2 * half[0] <= 0.075, (
                f"{name} half_x={half[0]} → width "
                f"{2*half[0]} > 0.075 m gripper margin"
            )

    def test_proxy_items_constant_is_populated(self):
        # At minimum we expect the items used in HAL access(-19) to be present.
        for required in ("meat_can", "mustard_bottle", "cracker_box"):
            assert required in YCB_PROXY_ITEMS
