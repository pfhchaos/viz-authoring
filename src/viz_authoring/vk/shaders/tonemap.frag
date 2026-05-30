#version 450
// Vulkan port of flame_sheep/rendering/shaders/tonemap.frag.
//
// Reads the histogram + max_buf, computes log-density alpha, maps the
// color_idx to a quick-and-dirty hue spectrum, outputs the final pixel
// color. The full GL version supports a palette texture, multi-monitor
// viewports, and a vibrancy parameter — this Phase 3 minimal cut skips
// them so the first end-to-end Vulkan render lands quickly. Each gap
// is a known follow-up.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(std430, binding = 0) readonly buffer Histogram {
    uint histogram[];
};

layout(std430, binding = 1) readonly buffer MaxBuf {
    uint max_hits;
};

layout(push_constant) uniform PushConstants {
    int width;
    int height;
    float gamma;
};

#define COLOR_SCALE 1000000u

// Tiny built-in palette: triadic spectrum (red→green→blue). Stand-in
// for the GL version's 256-bin sampler2D palette — gets us to a visible
// flame; real palette support lands in Phase 3.5.
vec3 simple_palette(float t) {
    t = clamp(t, 0.0, 1.0);
    // Smooth blend across three primary colors at t=0/0.5/1.
    vec3 c0 = vec3(1.0, 0.2, 0.05);   // warm red
    vec3 c1 = vec3(0.05, 0.9, 0.4);   // green
    vec3 c2 = vec3(0.1, 0.3, 1.0);    // blue
    if (t < 0.5) return mix(c0, c1, t * 2.0);
    else         return mix(c1, c2, (t - 0.5) * 2.0);
}

void main() {
    // Map UV → pixel coords in the histogram. UV [0,1]^2 covers the
    // full canvas; clamp to safe integer pixel index.
    int px = int(v_uv.x * float(width));
    int py = int(v_uv.y * float(height));
    px = clamp(px, 0, width - 1);
    py = clamp(py, 0, height - 1);

    uint n_pixels = uint(width * height);
    uint idx = uint(py * width + px);
    uint hits  = histogram[idx];
    uint color = histogram[n_pixels + idx];

    if (hits == 0u) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Log-density tone mapping. Without log, the histogram's huge
    // dynamic range (center ~millions of hits, edges ~1) makes
    // everything-not-center invisible.
    float log_hits = log(float(hits) + 1.0);
    float log_max  = log(float(max_hits) + 1.0);
    float alpha = (log_max > 0.0) ? clamp(log_hits / log_max, 0.0, 1.0)
                                   : 0.0;
    alpha = pow(alpha, 1.0 / gamma);

    // color_idx = color_acc / (hits * COLOR_SCALE)
    float color_idx = float(color) / (float(hits) * float(COLOR_SCALE));
    color_idx = clamp(color_idx, 0.0, 1.0);
    vec3 rgb = simple_palette(color_idx) * alpha;
    frag_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
