#version 450

// Three hardcoded vertices — no vertex buffer needed, the vertex shader
// produces positions + colors from gl_VertexIndex. Keeps the demo focused
// on instance/device/swapchain/pipeline lifecycle, not buffer plumbing.

layout(location = 0) out vec3 vColor;

vec2 positions[3] = vec2[](
    vec2( 0.0, -0.6),   // top
    vec2( 0.6,  0.6),   // bottom-right
    vec2(-0.6,  0.6)    // bottom-left
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.2, 0.2),
    vec3(0.2, 1.0, 0.2),
    vec3(0.2, 0.4, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    vColor = colors[gl_VertexIndex];
}
