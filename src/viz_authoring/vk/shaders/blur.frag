#version 450
// Vulkan port of flame_sheep/rendering/shaders/blur.frag.
//
// Single-axis 9-tap gaussian (sigma ≈ 2.0). Called twice per frame —
// once horizontal, once vertical. Same texture passed as input each
// pass, just to different framebuffers and with different direction.
//
// Vulkan adaptations: #version 450, sampler2D moves to a descriptor
// set, uniforms become push constants.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(push_constant) uniform PushConstants {
    vec2 u_direction;   // (1/w, 0) for horizontal, (0, 1/h) for vertical
    float u_radius;     // blur strength multiplier
};

void main() {
    // 9-tap gaussian weights (sigma ≈ 2.0, normalized to sum=1).
    // Same constants as the GL version so visual output matches.
    const float w0 = 0.227027;
    const float w1 = 0.1945946;
    const float w2 = 0.1216216;
    const float w3 = 0.054054;
    const float w4 = 0.016216;

    vec3 result = texture(u_texture, v_uv).rgb * w0;
    vec2 o;
    o = u_direction * 1.0 * u_radius;
    result += texture(u_texture, v_uv + o).rgb * w1;
    result += texture(u_texture, v_uv - o).rgb * w1;
    o = u_direction * 2.0 * u_radius;
    result += texture(u_texture, v_uv + o).rgb * w2;
    result += texture(u_texture, v_uv - o).rgb * w2;
    o = u_direction * 3.0 * u_radius;
    result += texture(u_texture, v_uv + o).rgb * w3;
    result += texture(u_texture, v_uv - o).rgb * w3;
    o = u_direction * 4.0 * u_radius;
    result += texture(u_texture, v_uv + o).rgb * w4;
    result += texture(u_texture, v_uv - o).rgb * w4;

    frag_color = vec4(result, 1.0);
}
