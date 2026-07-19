import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";

import { cn } from "../../lib/utils";

// The variant function is shared by links that visually match buttons.
// eslint-disable-next-line react-refresh/only-export-components
export const buttonVariants = cva(
  "inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-full border text-sm font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-45",
  {
    variants: {
      variant: {
        primary: "border-black bg-black px-6 text-white hover:bg-neutral-800",
        outline: "border-black bg-white px-6 text-black hover:bg-neutral-100",
        purple: "border-black bg-[var(--purple)] px-6 text-black hover:bg-[var(--purple-soft)]",
        green: "border-black bg-[var(--green)] px-6 text-black hover:bg-[var(--green-soft)]",
        ghost: "border-transparent bg-transparent px-4 text-black hover:bg-black/5",
        danger: "border-[var(--error)] bg-white px-6 text-[var(--error)] hover:bg-red-50",
      },
      size: {
        default: "h-12",
        small: "h-11 px-4",
        large: "h-14 px-8 text-base",
        icon: "size-11 p-0",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  },
);

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, size, type = "button", ...props }: ButtonProps) {
  return <button type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
