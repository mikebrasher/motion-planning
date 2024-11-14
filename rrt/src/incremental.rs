use glam::{Mat4, Vec4};
use std::sync::Arc;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    format::Format,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

/// A render pass which places an incoming image over frame filling it.
pub struct RenderPassIncremental {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    max_vertex: usize,
    idx_vertex: usize,
    device_buffer: Subbuffer<[MyVertex]>,
    uniform_buffer_allocator: SubbufferAllocator,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl RenderPassIncremental {
    pub fn new(
        gfx_queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        output_format: Format,
        swapchain_image_views: &[Arc<ImageView>],
        max_vertex: usize,
    ) -> RenderPassIncremental {
        let render_pass = vulkano::single_pass_renderpass!(
            gfx_queue.device().clone(),
            attachments: {
                color: {
                    format: output_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        // Before we draw, we have to create what is called a **pipeline**. A pipeline describes
        // how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing,
        // we create a **graphics** pipeline, but there are also other types of pipeline.
        let pipeline = {
            let device = gfx_queue.device();
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.
            let vs = vs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            // Automatically generate a vertex input state from the vertex shader's input
            // interface, that takes a single vertex buffer containing `Vertex` structs.
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            // We must now create a **pipeline layout** object, which describes the locations and
            // types of descriptor sets and push constants used by the shaders in the pipeline.
            //
            // Multiple pipelines can share a common layout object, which is more efficient. The
            // shaders in a pipeline must use a subset of the resources described in its pipeline
            // layout, but the pipeline layout is allowed to contain resources that are not present
            // in the shaders; they can be used by shaders in other pipelines that share the same
            // layout. Thus, it is a good idea to design shaders so that many pipelines have common
            // resource locations, which allows them to share pipeline layouts.
            let layout = PipelineLayout::new(
                device.clone(),
                // Since we only have one pipeline in this example, and thus one pipeline layout,
                // we automatically generate the creation info for it from the resources used in
                // the shaders. In a real application, you would specify this information manually
                // so that you can re-use one layout in multiple pipelines.
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(vertex_input_state),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::LineList,
                        ..Default::default()
                    }),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport, set to draw over the entire window.
                    viewport_state: Some(ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(RasterizationState {
                        line_width: 3.0,
                        ..Default::default()
                    }),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one,
                    // without any blending.
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    // Dynamic states allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            gfx_queue.device().clone(),
        ));

        let device_buffer: Subbuffer<[MyVertex]> = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            max_vertex as u64,
        )
        .unwrap();

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        RenderPassIncremental {
            gfx_queue,
            render_pass: render_pass.clone(),
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
            max_vertex,
            idx_vertex: 0,
            device_buffer,
            uniform_buffer_allocator,
            framebuffers: create_framebuffers(swapchain_image_views, render_pass),
        }
    }

    pub fn max_vertex(&self) -> usize {
        self.max_vertex
    }

    pub fn idx_vertex(&self) -> usize {
        self.idx_vertex
    }

    pub fn add_vertex(&mut self, x: f32, y: f32) {
        if self.idx_vertex < self.max_vertex {
            {
                let mut write_guard = self.device_buffer.write().unwrap();
                write_guard[self.idx_vertex] = MyVertex { position: [x, y] };
            }

            self.idx_vertex = self.idx_vertex + 1;
        }
    }

    /// Places the view exactly over the target swapchain image. The texture draw pipeline uses a
    /// quad onto which it places the view.
    pub fn render<F>(
        &self,
        before_future: F,
        target: Arc<ImageView>,
        image_index: u32,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        // Get dimensions.
        let img_dims: [u32; 2] = target.image().extent()[0..2].try_into().unwrap();

        // Create primary command buffer builder.
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let uniform_buffer: Subbuffer<vs::Data> = {
            let uniform_data = vs::Data {
                mvp: Mat4::from_cols(
                    Vec4::from((1.0, 0.0, 0.0, 0.0)),
                    Vec4::from((0.0, -1.0, 0.0, 0.0)),
                    Vec4::from((0.0, 0.0, -0.5, -1.0)),
                    Vec4::from((0.0, 0.0, 0.5, 1.0)),
                )
                .to_cols_array_2d(),
            };

            let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *buffer.write().unwrap() = uniform_data;

            buffer
        };

        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )
        .unwrap();

        // Begin render pass.
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.5, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                Default::default(),
            )
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [img_dims[0] as f32, img_dims[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            // .bind_vertex_buffers(0, vertex_buffer.clone())
            .bind_vertex_buffers(0, self.device_buffer.clone())
            .unwrap();

        unsafe { command_buffer_builder.draw(self.idx_vertex as u32, 1, 0, 0) }.unwrap();

        // End render pass.
        command_buffer_builder
            .end_render_pass(Default::default())
            .unwrap();

        // Build command buffer.
        let command_buffer = command_buffer_builder.build().unwrap();

        // Execute primary command buffer.
        let after_future = before_future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }

    pub fn recreate_framebuffers(&mut self, swapchain_image_views: &[Arc<ImageView>]) {
        self.framebuffers = create_framebuffers(swapchain_image_views, self.render_pass.clone());
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

fn create_framebuffers(
    swapchain_image_views: &[Arc<ImageView>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    swapchain_image_views
        .iter()
        .map(|swapchain_image_view| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![swapchain_image_view.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

// The next step is to create the shaders.
//
// The raw shader creation API provided by the vulkano library is unsafe for various
// reasons, so The `shader!` macro provides a way to generate a Rust module from GLSL
// source - in the example below, the source is provided as a string input directly to the
// shader, but a path to a source file can be provided as well. Note that the user must
// specify the type of shader (e.g. "vertex", "fragment", etc.) using the `ty` option of
// the macro.
//
// The items generated by the `shader!` macro include a `load` function which loads the
// shader using an input logical device. The module also includes type definitions for
// layout structures defined in the shader source, for example uniforms and push constants.
//
// A more detailed overview of what the `shader!` macro generates can be found in the
// vulkano-shaders crate docs. You can view them at https://docs.rs/vulkano-shaders/
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;

            // layout(set = 0, binding = 0) uniform Data {
            //     mat4 mvp;
            // } ubo;

            layout(set = 0, binding = 0) uniform Data {
                mat4 mvp;
            } ubo;

            void main() {
                gl_Position = ubo.mvp * vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
