"use client";
import { useId, useState } from "react";
import type { ComponentPropsWithRef, KeyboardEvent, ReactNode } from "react";

/** A password input with Show/Hide and a Caps Lock warning. `aside` sits at the end of the
 * label row (the sign-in page puts "Forgot password?" there). */
export function PasswordField({ label, aside, ...props }: ComponentPropsWithRef<"input"> & { label: string; aside?: ReactNode }) {
  const generated = useId();
  const id = props.id ?? generated;
  const [shown, setShown] = useState(false);
  const [caps, setCaps] = useState(false);
  const watchCaps = (e: KeyboardEvent<HTMLInputElement>) => {
    if (typeof e.getModifierState === "function") setCaps(e.getModifierState("CapsLock"));
  };
  return <div className="field">
    {aside ? <div className="field-label-row"><label htmlFor={id}>{label}</label>{aside}</div> : <label htmlFor={id}>{label}</label>}
    <div className="password-input">
      <input {...props} id={id} type={shown ? "text" : "password"}
        onKeyDown={(e) => { watchCaps(e); props.onKeyDown?.(e); }}
        onKeyUp={(e) => { watchCaps(e); props.onKeyUp?.(e); }}
        onBlur={(e) => { setCaps(false); props.onBlur?.(e); }} />
      <button type="button" className="btn btn-sm" aria-controls={id} aria-pressed={shown} aria-label={`${shown ? "Hide" : "Show"} ${label.toLowerCase()}`} onClick={() => setShown((v) => !v)}>{shown ? "Hide" : "Show"}</button>
    </div>
    {caps && <p className="caps-warn" role="status">Caps Lock is on</p>}
  </div>;
}
