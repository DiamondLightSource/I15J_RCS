import { useState, useEffect, useRef } from "react";
import { Container, Button, Typography } from "@mui/material";
import TrapezoidSelector from "../components/TrapezoidSelector";
import PointAnnotator from "../components/PointAnnotator";

const serverURL = "http://localhost:8000"

export interface Point {
    x: number;
    y: number;
}

export default function AnnotatePage() {
    const [stage, setStage] = useState<"table" | "centres" | "annotated">("table");

    const [ready, setReady] = useState(false);
    const [trapezoid, setTrapezoid] = useState(null);
    const [centres, setCentres] = useState([]);

    const [raw_image, setRawImage] = useState<string | null>(null);
    const [dewarped_image, setDewarpedImage] = useState<string | null>(null);
    const [annotated_image, setAnnotatedImage] = useState<string | null>(null);

    const [initial_dewarp, setInitialDewarp] = useState<Point[]>([
        { x: 0.1, y: 0.1 },   // top-left
        { x: 0.2, y: 0.1 },   // top-right
        { x: 0.25, y: 0.2 },  // bottom-right
        { x: 0.05, y: 0.2 },   // bottom-left
    ]);

    const imgRef = useRef<HTMLImageElement>(null);

    async function fetchInitialDewarp(endpoint: string): Promise<Point[]> {
        const res = await fetch(endpoint);
        return res.json() as Promise<Point[]>;
    }

    async function fetchImageUrl(endpoint: string): Promise<string> {
        const res = await fetch(endpoint);
        const blob = await res.blob();
        return URL.createObjectURL(blob);
    }

    useEffect(() => {
        async function load() {
            const imgUrl = await fetchImageUrl(serverURL + "/raw_image");
            setRawImage(imgUrl);

            await new Promise<void>((resolve) => {
                const img = new Image();
                img.src = imgUrl;
                img.onload = () => resolve();
            });

            const points = await fetchInitialDewarp(serverURL + "/dewarp_coordinates");

            if (points.length > 0) {
                setInitialDewarp(points);
            }

            setReady(true);
        }

        load();
    }, []);

    const submitTableTrapezoid = async () => {
        if (!trapezoid) return;

        const w = imgRef.current.clientWidth;
        const h = imgRef.current.clientHeight;

        const normalized = trapezoid.map(p => ({
            x: p.x / w,
            y: p.y / h,
        }));

        console.log("Submitting coordinates:", normalized);

        await fetch(serverURL + "/dewarp_coordinates", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(normalized),
        });

        // Load dewarped image after submission
        fetchImageUrl(serverURL + "/dewarped_image").then(setDewarpedImage);
        setStage("centres");
    };

    const submitPositionCentres = async () => {
        if (!centres) return;
        console.log("Final centres:", centres);
        await fetch(serverURL + "/position_centres", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(centres),
        });

        // Load annotated image after submission
        fetchImageUrl(serverURL + "/annotated_image").then(setAnnotatedImage);
        setStage("annotated");
    };

    return (
        <Container maxWidth="md" sx={{ py: 4 }}>
            {stage === "table" && ready && (
                <>
                    <Typography variant="h5">Draw the trapezoid around the table</Typography>
                    <TrapezoidSelector imageRef={imgRef} imageSrc={raw_image} onChange={setTrapezoid} initialPoints={initial_dewarp} />
                    <Button
                        variant="contained"
                        sx={{ mt: 2 }}
                        onClick={submitTableTrapezoid}
                    >
                        Submit trapezoid
                    </Button>
                </>
            )}

            {stage === "centres" && (
                <>
                    <Typography variant="h5">Click on the rough centres of all the positions</Typography>
                    <PointAnnotator
                        imageRef={imgRef}
                        imageSrc={dewarped_image}
                        points={centres}
                        onChange={setCentres}
                    />
                    <Button
                        variant="contained"
                        sx={{ mt: 2 }}
                        onClick={submitPositionCentres}
                    >
                        Submit centres
                    </Button>
                </>
            )}

            {stage === "annotated" && (
                <>
                    <Typography variant="h5">Below is the annotated image</Typography>
                    <img ref={imgRef} src={annotated_image} style={{ display: "block", maxWidth: "100%" }} />
                </>
            )}
        </Container>
    );
}
