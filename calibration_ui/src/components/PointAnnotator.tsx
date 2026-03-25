import { useState, useRef } from "react";
import { Box } from "@mui/material";

export interface Point {
    x: number;
    y: number;
}

export default function PointAnnotator({ imageRef, imageSrc, points, onChange }
    : {
        imageRef: React.RefObject<HTMLImageElement | null>;
        imageSrc: string | null;
        points: Point[];
        onChange?: (pts: Point[]) => void;
    }) {
    const [localPoints, setLocalPoints] = useState(points);
    const dragging = useRef(false);

    const update = (pts: Point[]) => {
        setLocalPoints(pts);
        onChange?.(pts);
    };

    const addPoint = (e: React.MouseEvent<HTMLDivElement>) => {
        if (dragging.current) {
            dragging.current = false;
            return;
        }
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        update([...localPoints, { x, y }]);
    };

    const startDrag = (index: number, startX: number, startY: number) => {
        dragging.current = true;

        const orig = localPoints[index];

        const move = (ev: MouseEvent) => {
            const dx = ev.clientX - startX;
            const dy = ev.clientY - startY;

            const updated = localPoints.map((p, i) =>
                i === index ? { x: orig.x + dx, y: orig.y + dy } : p
            );

            update(updated);
        };

        const up = () => {
            window.removeEventListener("mousemove", move);
            window.removeEventListener("mouseup", up);
        };

        window.addEventListener("mousemove", move);
        window.addEventListener("mouseup", up);
    };

    return (
        <Box
            sx={{ position: "relative", display: "inline-block" }}
            onClick={addPoint}
        >
            <img ref={imageRef} src={imageSrc ?? undefined} style={{ display: "block", maxWidth: "100%" }} />

            <svg
                style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                }}
            >
                {localPoints.map((p, i) => (
                    <circle
                        key={i}
                        cx={p.x}
                        cy={p.y}
                        r={7}
                        fill="#2196f3"
                        stroke="white"
                        strokeWidth={2}
                        style={{ cursor: "grab", pointerEvents: "all" }}
                        onMouseDown={(e) => {
                            e.stopPropagation();
                            startDrag(i, e.clientX, e.clientY);
                        }}
                    />
                ))}
            </svg>
        </Box>
    );
}
