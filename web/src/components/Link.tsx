"use client";

import NextLink from "next/link";
import { useState } from "react";
import type { ComponentProps } from "react";

type Props = ComponentProps<typeof NextLink>;

/** next/link that prefetches a page only when the pointer or focus reaches the link.
 *
 * By default Next prefetches every link as soon as it scrolls into view, so each page load
 * also fetched the nav bar and every stock row in sight (about a dozen page payloads and
 * server renders) whether or not they were opened. Hover, touch or keyboard focus still
 * prefetches ahead of the click, so moving around stays quick. Pass `prefetch` to choose. */
export default function Link({ prefetch, onMouseEnter, onTouchStart, onFocus, ...rest }: Props) {
  const [armed, setArmed] = useState(false);
  if (prefetch !== undefined) return <NextLink prefetch={prefetch} onMouseEnter={onMouseEnter} onTouchStart={onTouchStart} onFocus={onFocus} {...rest} />;
  const arm = () => { if (!armed) setArmed(true); };
  return (
    <NextLink
      {...rest}
      prefetch={armed ? null : false}
      onMouseEnter={(e) => { arm(); onMouseEnter?.(e); }}
      onTouchStart={(e) => { arm(); onTouchStart?.(e); }}
      onFocus={(e) => { arm(); onFocus?.(e); }}
    />
  );
}
