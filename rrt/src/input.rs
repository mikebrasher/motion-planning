use vulkano_util::window::WindowDescriptor;
use winit::{
    dpi::PhysicalSize,
    event::{KeyEvent, MouseScrollDelta, WindowEvent},
    keyboard::{Key, NamedKey},
};

pub struct InputState {
    pub window_size: [f32; 2],
    pub scroll_delta: f32,
    pub reset_step: bool,
    pub toggle_auto: bool,
}

impl InputState {
    pub fn new() -> InputState {
        InputState {
            window_size: [
                WindowDescriptor::default().width,
                WindowDescriptor::default().height,
            ],
            scroll_delta: 0.0,
            reset_step: false,
            toggle_auto: false,
        }
    }

    pub fn reset(&mut self) {
        self.scroll_delta = 0.0;
        self.reset_step = false;
        self.toggle_auto = false;
    }

    pub fn handle_input(&mut self, window_size: PhysicalSize<u32>, event: &WindowEvent) {
        self.window_size = window_size.into();

        match event {
            WindowEvent::KeyboardInput { event, .. } => self.on_keyboard_event(event),
            WindowEvent::MouseWheel { delta, .. } => self.on_mouse_wheel_event(delta),
            _ => {}
        }
    }

    fn on_keyboard_event(&mut self, event: &KeyEvent) {
        match event.logical_key.as_ref() {
            Key::Character("r") => self.reset_step = event.state.is_pressed(),
            Key::Named(NamedKey::Space) => self.toggle_auto = event.state.is_pressed(),
            _ => {}
        }
    }

    fn on_mouse_wheel_event(&mut self, delta: &MouseScrollDelta) {
        let change = match delta {
            MouseScrollDelta::LineDelta(_x, y) => *y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        self.scroll_delta += change;
    }
}
