import { useState, useEffect } from "react";
import { Box } from "@mui/material";

type Point = { x: number; y: number };

export default function TrapezoidSelector({ imageRef, imageSrc, onChange, initialPoints }) {
    const [points, setPoints] = useState<Point[]>(initialPoints);


    const updatePoint = (index: number, newPos: Point) => {
        const updated = points.map((p, i) => (i === index ? newPos : p));
        updatePoints(updated)
    };

    const updatePoints = (newPoints: Point[]) => {
        setPoints(newPoints);
        onChange?.(newPoints);
    };

    useEffect(() => {
        const img = imageRef.current;
        if (!img) return;

        const handleLoad = () => {
            const width = img.clientWidth;
            const height = img.clientHeight;

            const pixelPoints = initialPoints.map(p => ({
                x: p.x * width,
                y: p.y * height,
            }));

            updatePoints(pixelPoints);
        };

        // If the image is already loaded
        if (img.complete) {
            handleLoad();
        } else {
            img.onload = handleLoad;
        }
    }, [imageRef, initialPoints]);


    return (
        <Box sx={{ position: "relative", display: "inline-block" }}>
            <img
                ref={imageRef}
                src={imageSrc}
                style={{ display: "block", maxWidth: "100%" }}
            />

            {/* SVG overlay */}
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
                {/* Trapezoid polygon */}
                <polygon
                    points={points.map((p) => `${p.x},${p.y}`).join(" ")}
                    fill="rgba(255, 64, 129, 0.2)"
                    stroke="#ff4081"
                    strokeWidth={2}
                />

                {/* Draggable handles */}
                {points.map((p, i) => (
                    <circle
                        key={i}
                        cx={p.x}
                        cy={p.y}
                        r={8}
                        fill="#ff4081"
                        stroke="white"
                        strokeWidth={2}
                        style={{ cursor: "grab", pointerEvents: "all" }}
                        onMouseDown={(e) => {
                            e.preventDefault();
                            const startX = e.clientX;
                            const startY = e.clientY;

                            const move = (ev: MouseEvent) => {
                                const dx = ev.clientX - startX;
                                const dy = ev.clientY - startY;
                                updatePoint(i, { x: p.x + dx, y: p.y + dy });
                            };

                            const up = () => {
                                window.removeEventListener("mousemove", move);
                                window.removeEventListener("mouseup", up);
                            };

                            window.addEventListener("mousemove", move);
                            window.addEventListener("mouseup", up);
                        }}
                    />
                ))}
            </svg>
        </Box>
    );
}
