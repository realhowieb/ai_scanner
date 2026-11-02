import { describe, expect, it } from "vitest";

import { classifyPixels } from "@/components/TickerLogo";

/** A 48x48 RGBA buffer filled by `px(i)`. */
function img(px: (i: number) => [number, number, number, number]): Uint8ClampedArray {
  const d = new Uint8ClampedArray(48 * 48 * 4);
  for (let i = 0; i < 48 * 48; i += 1) d.set(px(i), i * 4);
  return d;
}

describe("classifyPixels", () => {
  it("keeps a dark logo on a cut-out background on the light chip", () => {
    expect(classifyPixels(img((i) => (i % 3 ? [20, 40, 160, 255] : [0, 0, 0, 0])))).toBe("standard");
  });
  it("moves a white cut-out logo to the dark chip", () => {
    expect(classifyPixels(img((i) => (i % 3 ? [250, 250, 250, 255] : [0, 0, 0, 0])))).toBe("light");
  });
  it("treats a busy opaque image as a photo (letter tile)", () => {
    expect(classifyPixels(img((i) => [(i * 7) % 256, ((i * 13) >> 2) % 256, ((i >> 4) * 41) % 256, 255]))).toBe("none");
  });
  it("keeps an opaque flat logo", () => {
    expect(classifyPixels(img((i) => (i % 2 ? [255, 255, 255, 255] : [200, 0, 0, 255])))).toBe("standard");
  });
  it("reads a fully transparent image as no logo", () => {
    expect(classifyPixels(img(() => [0, 0, 0, 0]))).toBe("none");
  });
});
