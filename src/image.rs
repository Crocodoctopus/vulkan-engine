use ash::vk;

#[derive(Debug)]
pub(crate) struct Image {
    pub image: vk::Image,
    pub alloc: vk_mem::Allocation,
}

#[derive(Debug)]
pub(crate) struct ImageView {
    pub view: vk::ImageView,
}
