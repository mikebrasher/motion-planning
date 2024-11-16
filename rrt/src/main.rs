use incremental::RenderPassIncremental;
use input::InputState;
use std::{error::Error, sync::Arc, time::Duration};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::DeviceFeatures,
    image::ImageUsage,
    swapchain::PresentMode,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::DEFAULT_IMAGE_FORMAT,
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

mod incremental;
mod input;

fn main() -> Result<(), impl Error> {
    // Create the event loop.
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    render_pass: RenderPassIncremental,
    input_state: InputState,
    idx_step: usize,
    max_step: usize,
}

impl App {
    fn new(_event_loop: &EventLoop<()>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: DeviceFeatures {
                wide_lines: true,
                ..Default::default()
            },
            ..Default::default()
        });
        let windows = VulkanoWindows::default();

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        App {
            context,
            windows,
            command_buffer_allocator,
            descriptor_set_allocator,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("Resumed");

        let _id = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: "RRT".to_string(),
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
            |_| {},
        );

        let render_target_id = 0;
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();

        // Make sure the image usage is correct (based on your pipeline).
        window_renderer.add_additional_image_view(
            render_target_id,
            DEFAULT_IMAGE_FORMAT,
            ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
        );

        let gfx_queue = self.context.graphics_queue();

        let max_step = 50;
        let max_vertex = 2 * max_step;

        let mut render_pass = RenderPassIncremental::new(
            gfx_queue.clone(),
            self.command_buffer_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            window_renderer.swapchain_format(),
            window_renderer.swapchain_image_views(),
            max_vertex,
        );

        let h = 1.0 / max_step as f32;
        for idx_step in 0..max_step {
            // add incremental vertex data
            let f = 2.0 * h * (idx_step as f32);
            let x = -1.0 + f;
            let y = 1.0 - h - f;
            render_pass.add_vertex(x, y);
            render_pass.add_vertex(x + h, y + h);
        }

        self.rcx = Some(RenderContext {
            render_pass,
            input_state: InputState::new(),
            max_step,
            idx_step: 0,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();
        let window_size = renderer.window().inner_size();

        match event {
            WindowEvent::CloseRequested => {
                println!("Close Requested");
                event_loop.exit();
            }
            WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                println!("Resized");
                renderer.resize();
            }
            WindowEvent::RedrawRequested => {
                println!("Redraw Requested");

                // Skip this frame when minimized.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.update_state_after_inputs();

                // Start the frame.
                let before_pipeline_future = match renderer.acquire(
                    Some(Duration::from_millis(1000)),
                    |swapchain_image_views| {
                        rcx.render_pass.recreate_framebuffers(swapchain_image_views)
                    },
                ) {
                    Err(e) => {
                        println!("{e}");
                        return;
                    }
                    Ok(future) => future,
                };

                // add incremental vertex data
                let h = 1.0 / rcx.render_pass.max_vertex() as f32;
                let f = 2.0 * h * (rcx.render_pass.idx_vertex() as f32);
                let x = -1.0 + f;
                let y = 1.0 - h - f;
                rcx.render_pass.add_vertex(x, y);
                rcx.render_pass.add_vertex(x + h, y + h);

                // Render the image over the swapchain image, inputting the previous future.
                let after_renderpass_future = rcx.render_pass.render(
                    before_pipeline_future,
                    renderer.swapchain_image_view(),
                    renderer.image_index(),
                    2 * rcx.idx_step as u32,
                );

                // Finish the frame (which presents the view), inputting the last future. Wait for
                // the future so resources are not in use when we render.
                renderer.present(after_renderpass_future, true);

                rcx.input_state.reset();
            }
            _ => {
                // Pass event for the app to handle our inputs.
                rcx.input_state.handle_input(window_size, &event);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        println!("About to Wait");
        self.windows
            .get_primary_renderer_mut()
            .unwrap()
            .window()
            .request_redraw();
    }
}

impl RenderContext {
    fn update_state_after_inputs(&mut self) {
        self.idx_step = if self.input_state.scroll_delta > 0.0 {
            if self.idx_step < self.max_step - 1 {
                self.idx_step + 1
            } else {
                self.idx_step
            }
        } else if self.input_state.scroll_delta < 0.0 {
            if self.idx_step > 0 {
                self.idx_step - 1
            } else {
                self.idx_step
            }
        } else {
            self.idx_step
        };

        if self.input_state.reset_step {
            println!("Reset step = 0");
            self.idx_step = 0;
        }

        println!("Update step = {}", self.idx_step);
    }
}
