use vulkano_util::window::WindowDescriptor;
use winit::{dpi::PhysicalSize, event::WindowEvent};

pub struct InputState {
    pub window_size: [f32; 2],
}

impl InputState {
    pub fn new() -> InputState {
        InputState {
            window_size: [
                WindowDescriptor::default().width,
                WindowDescriptor::default().height,
            ],
        }
    }

    pub fn handle_input(&mut self, window_size: PhysicalSize<u32>, event: &WindowEvent) {
        self.window_size = window_size.into();

        match event {
            // WindowEvent::KeyboardInput { event, .. } => self.on_keyboard_event(event),
            // WindowEvent::MouseInput { state, button, .. } => {
            //     self.on_mouse_click_event(*state, *button)
            // }
            // WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved_event(position),
            // WindowEvent::MouseWheel { delta, .. } => self.on_mouse_wheel_event(delta),
            _ => {}
        }
    }
}
