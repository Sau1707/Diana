import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";
import type { ReactNode } from "react";

export function Accordion({ children }: { children: ReactNode }) {
  return <AccordionPrimitive.Root type="multiple" className="divide-y divide-[var(--border)] border-y border-[var(--border)]">{children}</AccordionPrimitive.Root>;
}

export function AccordionItem({ value, title, children }: { value: string; title: string; children: ReactNode }) {
  return (
    <AccordionPrimitive.Item value={value}>
      <AccordionPrimitive.Header>
        <AccordionPrimitive.Trigger className="group flex min-h-14 w-full items-center justify-between gap-4 py-3 text-left font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black">
          {title}
          <ChevronDown className="size-4 transition-transform group-data-[state=open]:rotate-180" aria-hidden="true" />
        </AccordionPrimitive.Trigger>
      </AccordionPrimitive.Header>
      <AccordionPrimitive.Content className="overflow-hidden pb-4 pr-8 text-sm leading-6 text-[var(--muted)] data-[state=open]:animate-in">
        {children}
      </AccordionPrimitive.Content>
    </AccordionPrimitive.Item>
  );
}
