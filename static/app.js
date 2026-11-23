const $ = (id) =>
    document.getElementById(id);


let selectedFile = null;
let session = null;
let currentResult = null;
const PUBLIC_DEMO = window.HEALTHYAI_PUBLIC_DEMO === true;

const classNames = {

    akiec:
        "Actinic keratoses",

    bcc:
        "Basal cell carcinoma",

    bkl:
        "Benign keratosis-like lesion",

    df:
        "Dermatofibroma",

    mel:
        "Melanoma",

    nv:
        "Melanocytic nevus",

    vasc:
        "Vascular lesion",
};


function prettyLabel(label) {

    return (
        classNames[label]
        || label
    );
}


function showError(message) {

    const box = $(
        "errorBox"
    );

    box.textContent =
        message;

    box.classList.remove(
        "hidden"
    );
}


function clearError() {

    $(
        "errorBox"
    ).classList.add(
        "hidden"
    );
}


function formatDate(value) {

    try {

        return new Date(
            value
        ).toLocaleString();

    } catch {

        return value;

    }
}


function escapeHtml(value) {

    return String(value)
        .replace(
            /[&<>'"]/g,
            character => {

                const replacements = {

                    "&":
                        "&amp;",

                    "<":
                        "&lt;",

                    ">":
                        "&gt;",

                    "'":
                        "&#39;",

                    '"':
                        "&quot;",
                };

                return replacements[
                    character
                ];
            }
        );
}


// ============================================================
// LOAD MODEL
// ============================================================


async function loadModel() {
    try {
        const response = await fetch("/api/health");

        if (!response.ok) {
            throw new Error(
                `Health API failed: HTTP ${response.status}`
            );
        }

        const health = await response.json();

        if (!health.model || !health.model.ready) {
            $("modelStatus").textContent = "Model not installed";

            showError(
                "The trained ONNX model is not available on the server."
            );
            return;
        }

        if (!window.ort) {
            throw new Error(
                "ONNX Runtime Web is not loaded. Check ort.min.js."
            );
        }

        // ONNX Runtime Web configuration
        // Must run BEFORE creating the inference session.
        ort.env.wasm.wasmPaths = "/static/vendor/";
        ort.env.wasm.numThreads = 1;

        console.log("Loading HealthyAI ONNX model...");

        const modelUrl =
            `/model/${encodeURIComponent(health.model.model)}`;

        session = await ort.InferenceSession.create(
            modelUrl,
            {
                executionProviders: ["wasm"]
            }
        );

        $("modelStatus").textContent =
            `Model ready · ${health.model.size_mb} MB`;

        clearError();

        console.log("HealthyAI model loaded successfully!");

    } catch (error) {
        session = null;

        $("modelStatus").textContent = "Model unavailable";

        console.error(
            "HealthyAI model loading error:",
            error
        );

        showError(
            error.message || "Could not load the ONNX model."
        );
    }
}


function imageToTensor(file) {

    return new Promise(
        (resolve, reject) => {

            const image =
                new Image();


            image.onload = () => {

                const canvas =
                    document.createElement(
                        "canvas"
                    );

                canvas.width =
                    224;

                canvas.height =
                    224;


                const context =
                    canvas.getContext(
                        "2d"
                    );


                /*
                 * Center crop.
                 */

                const scale =
                    Math.max(
                        224 / image.width,
                        224 / image.height
                    );


                const width =
                    image.width
                    * scale;


                const height =
                    image.height
                    * scale;


                const x =
                    (224 - width)
                    / 2;


                const y =
                    (224 - height)
                    / 2;


                context.drawImage(
                    image,
                    x,
                    y,
                    width,
                    height
                );


                const pixels =
                    context.getImageData(
                        0,
                        0,
                        224,
                        224
                    ).data;


                const data =
                    new Float32Array(
                        3
                        * 224
                        * 224
                    );


                const imageArea =
                    224 * 224;


                for (
                    let y = 0;
                    y < 224;
                    y++
                ) {

                    for (
                        let x = 0;
                        x < 224;
                        x++
                    ) {

                        const pixelIndex =
                            (
                                y * 224
                                + x
                            ) * 4;


                        const red =
                            pixels[
                                pixelIndex
                            ] / 255;


                        const green =
                            pixels[
                                pixelIndex + 1
                            ] / 255;


                        const blue =
                            pixels[
                                pixelIndex + 2
                            ] / 255;


                        const index =
                            y * 224 + x;


                        /*
                         * ImageNet normalization.
                         */

                        data[
                            index
                        ] =
                            (
                                red
                                - 0.485
                            )
                            / 0.229;


                        data[
                            imageArea
                            + index
                        ] =
                            (
                                green
                                - 0.456
                            )
                            / 0.224;


                        data[
                            imageArea * 2
                            + index
                        ] =
                            (
                                blue
                                - 0.406
                            )
                            / 0.225;
                    }
                }


                resolve({

                    tensor:
                        new ort.Tensor(
                            "float32",
                            data,
                            [
                                1,
                                3,
                                224,
                                224,
                            ]
                        ),

                    width:
                        image.width,

                    height:
                        image.height,
                });
            };


            image.onerror = () => {

                reject(
                    new Error(
                        "Could not decode the image."
                    )
                );
            };


            image.src =
                URL.createObjectURL(
                    file
                );
        }
    );
}


// ============================================================
// PREVIEW
// ============================================================

function showPreview(file) {

    const url =
        URL.createObjectURL(
            file
        );

    $(
        "preview"
    ).src = url;

    $(
        "dropZone"
    ).classList.add(
        "has-image"
    );
}


// ============================================================
// SOFTMAX
// ============================================================

function softmax(values) {

    const maximum =
        Math.max(
            ...values
        );


    const exponentials =
        values.map(
            value =>
                Math.exp(
                    value
                    - maximum
                )
        );


    const total =
        exponentials.reduce(
            (
                sum,
                value
            ) =>
                sum + value,
            0
        );


    return exponentials.map(
        value =>
            value / total
    );
}


// ============================================================
// SCREENING BAND
// ============================================================

function getBand(
    label,
    confidence
) {

    if (
        confidence < 0.60
    ) {

        return "UNCERTAIN";
    }


    if (
        [
            "mel",
            "bcc",
            "akiec",
        ].includes(label)
    ) {

        return "REVIEW";
    }


    return "LOWER-CONCERN";
}


// ============================================================
// ANALYZE
// ============================================================

async function analyze() {

    clearError();


    if (
        !selectedFile
    ) {

        showError(
            "Choose an image first."
        );

        return;
    }


    if (!session) {

        showError(
            "The trained ONNX model is not ready."
        );

        return;
    }


    const button =
        $(
            "analyzeBtn"
        );


    button.disabled =
        true;

    button.textContent =
        "Analyzing…";


    try {

        const {
            tensor,
            width,
            height,
        } =
            await imageToTensor(
                selectedFile
            );


        const outputs =
            await session.run(
                {
                    [
                        session.inputNames[0]
                    ]:
                        tensor,
                }
            );


        const output =
            outputs[
                session.outputNames[0]
            ];


        const logits =
            Array.from(
                output.data
            );


        const probabilities =
            softmax(
                logits
            );


        const health =
            await fetch(
                "/api/health"
            ).then(
                response =>
                    response.json()
            );


        const classes =
            health.model.classes;


        const ranked =
            probabilities
                .map(
                    (
                        probability,
                        index
                    ) => ({

                        label:
                            classes[
                                index
                            ]
                            || String(
                                index
                            ),

                        probability,
                    })
                )
                .sort(
                    (
                        a,
                        b
                    ) =>
                        b.probability
                        - a.probability
                );


        const best =
            ranked[0];


        const band =
            getBand(
                best.label,
                best.probability
            );


        currentResult = {

            filename:
                selectedFile.name,

            label:
                best.label,

            confidence:
                best.probability,

            top_predictions:
                ranked.slice(
                    0,
                    3
                ),

            image_width:
                width,

            image_height:
                height,

            image_format:
                selectedFile.type,
        };


        /*
         * Display result.
         */

        $(
            "emptyResult"
        ).classList.add(
            "hidden"
        );


        $(
            "result"
        ).classList.remove(
            "hidden"
        );


        $(
            "label"
        ).textContent =
            prettyLabel(
                best.label
            );


        $(
            "confidence"
        ).textContent =
            `${
                (
                    best.probability
                    * 100
                ).toFixed(1)
            }%`;


        $(
            "confidenceBar"
        ).style.width =
            `${
                best.probability
                * 100
            }%`;


        const bandElement =
            $(
                "band"
            );


        bandElement.textContent =
            band;


        bandElement.className =
            "band " +
            (
                band === "REVIEW"
                    ? "review"
                    : band === "UNCERTAIN"
                        ? "uncertain"
                        : "lower"
            );


        $(
            "uncertainty"
        ).textContent =
            best.probability < 0.60
                ? "The model is not confident enough to show a stronger project-level screening band."
                : "This is a machine-learning output, not a clinical diagnosis.";


        $(
            "topPredictions"
        ).innerHTML =
            ranked
                .slice(
                    0,
                    3
                )
                .map(
                    prediction => {

                        const percentage =
                            (
                                prediction.probability
                                * 100
                            ).toFixed(
                                1
                            );


                        return `
                            <div class="pred">

                                <div class="pred-label">
                                    ${escapeHtml(
                                        prettyLabel(
                                            prediction.label
                                        )
                                    )}
                                </div>

                                <div class="pred-bar">
                                    <i
                                        style="width:${percentage}%"
                                    ></i>
                                </div>

                                <div class="pred-val">
                                    ${percentage}%
                                </div>

                            </div>
                        `;
                    }
                )
                .join("");


    } catch (error) {

        showError(
            error.message
            || "Analysis failed."
        );

    } finally {

        button.disabled =
            false;

        button.textContent =
            "Analyze image";
    }
}


// ============================================================
// SAVE HISTORY
// ============================================================

async function saveHistory() {
    if (PUBLIC_DEMO) return;

    if (!currentResult) {

        showError(
            "Analyze an image first."
        );

        return;
    }


    try {

        const response =
            await fetch(
                "/api/history",
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body:
                        JSON.stringify(
                            currentResult
                        ),
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error
                || "Could not save history."
            );
        }


        await loadHistory();
        await loadStats();


        $(
            "historyBtn"
        ).textContent =
            "Saved ✓";


    } catch (error) {

        showError(
            error.message
        );
    }
}


// ============================================================
// PDF REPORT
// ============================================================

async function downloadReport() {

    if (!currentResult) {

        showError(
            "Analyze an image first."
        );

        return;
    }


    try {

        const response =
            await fetch(
                "/api/report",
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body:
                        JSON.stringify(
                            currentResult
                        ),
                }
            );


        if (!response.ok) {

            const data =
                await response.json();

            throw new Error(
                data.error
                || "Could not generate report."
            );
        }


        const blob =
            await response.blob();


        const url =
            URL.createObjectURL(
                blob
            );


        const link =
            document.createElement(
                "a"
            );


        link.href =
            url;

        link.download =
            "healthy_ai_report.pdf";


        document.body.appendChild(
            link
        );

        link.click();

        link.remove();

        URL.revokeObjectURL(
            url
        );


    } catch (error) {

        showError(
            error.message
        );
    }
}


// ============================================================
// HISTORY
// ============================================================

async function loadHistory() {
    if (PUBLIC_DEMO) {
        const body = $("historyBody");

        if (body) {
            body.innerHTML = `
                <tr>
                    <td colspan="5" class="empty-cell">
                        Prediction history is unavailable in this public demonstration.
                        You can still analyze images and download PDF reports.
                    </td>
                </tr>
            `;
        }

        return;
    }

    try {

        const response =
            await fetch(
                "/api/history"
            );


        const rows =
            await response.json();


        const body =
            $(
                "historyBody"
            );


        if (
            !rows.length
        ) {

            body.innerHTML =
                `
                <tr>
                    <td
                        colspan="5"
                        class="empty-cell"
                    >
                        No saved screenings.
                    </td>
                </tr>
                `;

            return;
        }


        body.innerHTML =
            rows
                .map(
                    row => {

                        const className =
                            row.screening_band === "REVIEW"
                                ? "review-text"
                                : row.screening_band === "UNCERTAIN"
                                    ? "uncertain-text"
                                    : "lower-text";


                        return `
                            <tr>

                                <td>
                                    ${escapeHtml(
                                        formatDate(
                                            row.created_at
                                        )
                                    )}
                                </td>

                                <td>
                                    ${escapeHtml(
                                        row.filename
                                    )}
                                </td>

                                <td>
                                    ${escapeHtml(
                                        prettyLabel(
                                            row.label
                                        )
                                    )}
                                </td>

                                <td>
                                    ${row.confidence_percent}%
                                </td>

                                <td class="${className}">
                                    ${escapeHtml(
                                        row.screening_band
                                    )}
                                </td>

                            </tr>
                        `;
                    }
                )
                .join("");


    } catch (error) {

        console.error(
            "History error:",
            error
        );
    }
}


// ============================================================
// STATS
// ============================================================


async function loadStats() {
    const totalEl = $("totalStat");
    const avgEl = $("avgStat");
    const reviewEl = $("reviewStat");

    // Safe defaults for public-demo mode
    totalEl.textContent = "0";
    avgEl.textContent = "N/A";
    reviewEl.textContent = "0";

    try {
        const response = await fetch("/api/stats");

        if (!response.ok) {
            console.warn("Statistics unavailable:", response.status);
            return;
        }

        const stats = await response.json();

        totalEl.textContent = stats.total ?? 0;

        avgEl.textContent =
            stats.average_confidence != null
                ? `${stats.average_confidence}%`
                : "N/A";

        reviewEl.textContent =
            stats.bands?.REVIEW ?? 0;

    } catch (error) {
        console.error("Stats error:", error);
    }
}



// ============================================================
// FILE SELECTION
// ============================================================

$(
    "imageInput"
).addEventListener(
    "change",
    event => {

        const file =
            event.target.files[0];


        if (!file) {
            return;
        }


        selectedFile =
            file;


        showPreview(
            file
        );


        $(
            "analyzeBtn"
        ).disabled =
            false;


        clearError();
    }
);


// ============================================================
// DRAG & DROP
// ============================================================

$(
    "dropZone"
).addEventListener(
    "dragover",
    event => {

        event.preventDefault();

        event.currentTarget.classList.add(
            "dragging"
        );
    }
);


$(
    "dropZone"
).addEventListener(
    "dragleave",
    event => {

        event.currentTarget.classList.remove(
            "dragging"
        );
    }
);


$(
    "dropZone"
).addEventListener(
    "drop",
    event => {

        event.preventDefault();

        event.currentTarget.classList.remove(
            "dragging"
        );


        const file =
            event.dataTransfer.files[0];


        if (!file) {
            return;
        }


        selectedFile =
            file;


        showPreview(
            file
        );


        $(
            "analyzeBtn"
        ).disabled =
            false;


        clearError();
    }
);


// ============================================================
// BUTTONS
// ============================================================

$(
    "analyzeBtn"
).addEventListener(
    "click",
    analyze
);


$(
    "historyBtn"
).addEventListener(
    "click",
    saveHistory
);


$(
    "reportBtn"
).addEventListener(
    "click",
    downloadReport
);


$(
    "clearBtn"
).addEventListener(
    "click",
    () => {

        selectedFile =
            null;

        currentResult =
            null;

        $(
            "imageInput"
        ).value = "";


        $(
            "preview"
        ).src = "";


        $(
            "dropZone"
        ).classList.remove(
            "has-image"
        );


        $(
            "analyzeBtn"
        ).disabled =
            true;


        $(
            "result"
        ).classList.add(
            "hidden"
        );


        $(
            "emptyResult"
        ).classList.remove(
            "hidden"
        );


        $(
            "band"
        ).textContent =
            "WAITING";


        $(
            "band"
        ).className =
            "band neutral";


        $(
            "historyBtn"
        ).textContent =
            "Save result";


        clearError();
    }
);


$(
    "clearHistory"
).addEventListener(
    "click",
    async () => {
        if (PUBLIC_DEMO) return;


        const confirmed =
            confirm(
                "Delete all saved screening records?"
            );


        if (!confirmed) {
            return;
        }


        await fetch(
            "/api/history",
            {
                method:
                    "DELETE",
            }
        );


        await loadHistory();
        await loadStats();
    }
);


// ============================================================
// START
// ============================================================

loadModel();

if (PUBLIC_DEMO) {
    const clearButton = $("clearHistory");
    if (clearButton) {
        clearButton.style.display = "none";
    }
    loadHistory();

    // Disable saving shared history in the public demo.
    const saveButton = $("historyBtn");

    if (saveButton) {
        saveButton.hidden = true;
        saveButton.style.display = "none";
    }

    // Show safe defaults when shared statistics are unavailable.
    $("totalStat").textContent = "N/A";
    $("avgStat").textContent = "N/A";
    $("reviewStat").textContent = "N/A";

} else {
    loadHistory();
    loadStats();
}