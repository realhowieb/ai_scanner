"use client";
import { useId, useState } from "react";
import type { InputHTMLAttributes } from "react";
export function PasswordField({ label, ...props }: InputHTMLAttributes<HTMLInputElement> & { label: string }) {
  const generated = useId();
  const id = props.id ?? generated;
  const [shown, setShown] = useState(false);
  return <div className="field"><label htmlFor={id}>{label}</label><div className="password-input">
    <input {...props} id={id} type={shown ? "text" : "password"} />
    <button type="button" className="btn btn-sm" aria-controls={id} aria-pressed={shown} aria-label={`${shown ? "Hide" : "Show"} ${label.toLowerCase()}`} onClick={() => setShown((v) => !v)}>{shown ? "Hide" : "Show"}</button>
  </div></div>;
}
