//! Shared lightmap slots: one slot per atlas, with one owner per mesh.
//! UV rectangles and sampling flags stay in RenderLightmap, never in this table.
use super::{LightmapSlabIndex, LightmapSlotIndex};
use bevy_asset::AssetId;
use bevy_image::Image;
use bevy_platform::collections::HashMap;

type Slot = (LightmapSlabIndex, LightmapSlotIndex);

pub(super) struct SharedSlots {
    enabled: bool,
    images: HashMap<AssetId<Image>, (Slot, usize)>,
}

impl Default for SharedSlots {
    fn default() -> Self {
        Self::new(true)
    }
}

impl SharedSlots {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            images: HashMap::default(),
        }
    }

    pub(super) fn acquire(&mut self, image: AssetId<Image>) -> Option<Slot> {
        if !self.enabled {
            return None;
        }
        let (slot, users) = self.images.get_mut(&image)?;
        *users += 1;
        Some(*slot)
    }

    pub(super) fn insert(&mut self, image: AssetId<Image>, slot: Slot) {
        if self.enabled {
            assert!(self.images.insert(image, (slot, 1)).is_none());
        }
    }

    /// Only the last owner frees a GPU binding or cancels its pending upload.
    pub(super) fn release(&mut self, image: AssetId<Image>, slot: Slot) -> Option<Slot> {
        if !self.enabled {
            return Some(slot);
        }
        let (_, users) = self
            .images
            .get_mut(&image)
            .expect("lightmap owner must have a slot");
        *users -= 1;
        if *users != 0 {
            return None;
        }
        self.images.remove(&image).map(|(slot, _)| slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(n: u128) -> AssetId<Image> {
        AssetId::Uuid {
            uuid: bevy_asset::uuid::Uuid::from_u128(n),
        }
    }

    fn slot(n: usize) -> Slot {
        (LightmapSlabIndex::from(n), LightmapSlotIndex::from(0_u32))
    }

    fn allocate(slots: &mut SharedSlots, image: AssetId<Image>, allocations: &mut usize) -> Slot {
        if let Some(slot) = slots.acquire(image) {
            return slot;
        }
        let slot = slot(*allocations);
        *allocations += 1;
        slots.insert(image, slot);
        slot
    }

    #[test]
    fn diagnostic_unshared_mode_allocates_and_releases_each_receiver() {
        let mut slots = SharedSlots::new(false);
        let mut allocations = 0;
        assert_eq!(allocate(&mut slots, image(1), &mut allocations), slot(0));
        assert_eq!(allocate(&mut slots, image(1), &mut allocations), slot(1));
        assert_eq!(allocations, 2);
        assert_eq!(slots.release(image(1), slot(0)), Some(slot(0)));
        assert_eq!(slots.release(image(1), slot(1)), Some(slot(1)));
    }

    #[test]
    fn shared_atlas_1462_meshes_allocate_four_slots_and_free_only_after_last_owner() {
        let mut slots = SharedSlots::default();
        let mut allocations = 0;
        for mesh in 0..1462 {
            let atlas = mesh % 4;
            assert_eq!(
                allocate(&mut slots, image(atlas), &mut allocations),
                slot(atlas as usize)
            );
        }
        assert_eq!(allocations, 4);
        for mesh in 0..1458 {
            assert_eq!(
                slots.release(image(mesh % 4), slot((mesh % 4) as usize)),
                None
            );
        }
        for mesh in 1458..1462 {
            assert_eq!(
                slots.release(image(mesh % 4), slot((mesh % 4) as usize)),
                Some(slot((mesh % 4) as usize))
            );
        }
        assert!(slots.images.is_empty());
    }

    #[test]
    fn replacing_one_atlas_user_preserves_other_users() {
        let mut slots = SharedSlots::default();
        let mut allocations = 0;
        let old = allocate(&mut slots, image(1), &mut allocations);
        assert_eq!(allocate(&mut slots, image(1), &mut allocations), old);
        assert_eq!(slots.release(image(1), slot(0)), None);
        let new = allocate(&mut slots, image(2), &mut allocations);
        assert_ne!(old, new);
        assert_eq!(slots.release(image(2), new), Some(new));
        assert_eq!(slots.release(image(1), slot(0)), Some(old));
    }

    #[test]
    fn final_release_allows_reload_into_recycled_slot() {
        let mut slots = SharedSlots::default();
        slots.insert(image(1), slot(0));
        assert_eq!(slots.release(image(1), slot(0)), Some(slot(0)));
        assert_eq!(slots.acquire(image(1)), None);
        slots.insert(image(2), slot(0));
        assert_eq!(slots.acquire(image(2)), Some(slot(0)));
        assert_eq!(slots.release(image(2), slot(0)), None);
        assert_eq!(slots.release(image(2), slot(0)), Some(slot(0)));
    }
}

#[cfg(test)]
mod mesh_parameter_tests {
    use super::super::RenderLightmap;
    use super::*;
    use bevy_math::Rect;

    #[test]
    fn atlas_slot_sharing_preserves_each_mesh_uv_rectangle_and_filtering() {
        let image = AssetId::<Image>::default();
        let slab = LightmapSlabIndex::from(0_usize);
        let slot = LightmapSlotIndex::from(0_u32);
        let first_rect = Rect::new(0.0, 0.0, 0.5, 0.5);
        let second_rect = Rect::new(0.5, 0.5, 1.0, 1.0);
        let first = RenderLightmap::new(first_rect, slab, slot, false, image);
        let second = RenderLightmap::new(second_rect, slab, slot, true, image);
        assert_eq!(
            (first.slab_index, first.slot_index),
            (second.slab_index, second.slot_index)
        );
        assert_eq!(first.uv_rect, first_rect);
        assert_eq!(second.uv_rect, second_rect);
        assert!(!first.bicubic_sampling);
        assert!(second.bicubic_sampling);
    }
}
