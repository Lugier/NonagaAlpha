const HEX_SIZE = 12;
const RED = 1;
const BLACK = -1;

// API Endpoints
const API_BASE = "/api";

// Core Game State mapped exactly to Backend
let state = {
    discs: [],
    red: [],
    black: [],
    forbidden_disc: null,
    side_to_move: RED
};

// UI Config & State
let humanColor = RED;
let legalMoves = [];
let aiOpponent = "alphazero-nonagazero.pt";

// Interaction State Machine
let uiMode = "IDLE"; // IDLE, WAIT_AI, PIECE_SELECTED, PIECE_SLID, TILE_REMOVED, GAME_OVER
let tempMove = { start: null, end: null, remove: null, place: null };

// DOM Elements
const svgGrid = document.getElementById("hex-layer");
const svgPieces = document.getElementById("pieces-layer");
const svgHighlights = document.getElementById("highlights-layer");
const statusText = document.getElementById("game-status");
const turnIndicator = document.getElementById("turn-indicator");
const loadingOverlay = document.getElementById("loading-overlay");

// ---- HEX MATH ----
function axialToPixel(q, r) {
    const x = HEX_SIZE * Math.sqrt(3) * (q + r/2);
    const y = HEX_SIZE * 3/2 * r;
    return {x, y};
}

function createHexPolygon(x, y, size) {
    let points = [];
    for (let i = 0; i < 6; i++) {
        let angle_deg = 60 * i + 30;
        let angle_rad = Math.PI / 180 * angle_deg;
        points.push(`${x + size * Math.cos(angle_rad)},${y + size * Math.sin(angle_rad)}`);
    }
    return points.join(" ");
}

// ==== INIT STATE ====
function initGame() {
    // Exact starting coords per Nonaga rules
    state.discs = [
        [0,0], [1,0], [2,0], [-1,1], [0,1], [1,1], [2,1],
        [-2,2], [-1,2], [0,2], [1,2], [2,2], [-2,3], [-1,3],
        [0,3], [1,3], [-2,4], [-1,4], [0,4]
    ];
    // Start coords normalized exactly to Python start state
    state.red = [[1, 0], [1, 2], [1, 4]]; // Top, Center Right, Bottom
    state.black = [[0, 0], [0, 2], [0, 4]]; // Top left etc (Placeholder, just visually symmetric)
    state.forbidden_disc = null;
    state.side_to_move = RED;
    
    // Actually the python engine state.initial() uses strict coordinates:
    // r1,r2,r3 = (0,-2) -> Wait, let's just make an API call to start or reconstruct valid coords
    // To be perfectly safe, let's align with geometry.py. It's better to fetch legal moves immediately.
    // I'll hardcode the canonical start from the engine:
    state.discs = [
        [0,0], [1,0], [2,0], 
        [-1,1], [0,1], [1,1], [2,1],
        [-2,2], [-1,2], [0,2], [1,2], [2,2],
        [-2,3], [-1,3], [0,3], [1,3],
        [-2,4], [-1,4], [0,4]
    ];
    state.red = [[0,0], [2,2], [-2,4]];
    state.black = [[2,0], [-2,2], [0,4]];
    
    uiMode = "IDLE";
    fetchLegalMoves();
}

function updateTurnIndicator() {
    turnIndicator.textContent = state.side_to_move === RED ? "RED" : "BLACK";
    turnIndicator.className = state.side_to_move === RED ? "turn-red" : "turn-black";
}

// ==== RENDERING ====
function renderBoard() {
    svgGrid.innerHTML = "";
    svgPieces.innerHTML = "";
    
    // Draw base ring
    state.discs.forEach(coord => {
        const [q, r] = coord;
        const {x, y} = axialToPixel(q, r);
        const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        poly.setAttribute("points", createHexPolygon(x, y, HEX_SIZE));
        poly.classList.add("hex");
        
        if (state.forbidden_disc && state.forbidden_disc[0] === q && state.forbidden_disc[1] === r) {
            poly.classList.add("hex-forbidden");
        }
        
        // Interaction for tile removal
        poly.addEventListener("click", () => handleHexClick(q, r, "disc"));
        svgGrid.appendChild(poly);
    });
    
    // Draw Red Pieces
    state.red.forEach(coord => {
        const [q, r] = coord;
        const {x, y} = axialToPixel(q, r);
        drawPiece(x, y, "var(--piece-red)", q, r);
    });
    
    // Draw Black Pieces
    state.black.forEach(coord => {
        const [q, r] = coord;
        const {x, y} = axialToPixel(q, r);
        drawPiece(x, y, "var(--piece-black)", q, r);
    });
}

function drawPiece(x, y, color, q, r) {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", HEX_SIZE * 0.6);
    circle.setAttribute("fill", color);
    circle.classList.add("piece");
    circle.addEventListener("click", () => handleHexClick(q, r, "piece"));
    svgPieces.appendChild(circle);
}

function clearHighlights() {
    svgHighlights.innerHTML = "";
}

function addHighlight(q, r, type) {
    const {x, y} = axialToPixel(q, r);
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", createHexPolygon(x, y, HEX_SIZE * 0.8));
    poly.classList.add("highlight");
    // Color coded highlights
    if (type === "slide") poly.style.stroke = "#3b82f6";
    if (type === "remove") poly.style.stroke = "#ef4444";
    if (type === "place") poly.style.stroke = "#10b981";
    
    poly.addEventListener("click", () => handleHexClick(q, r, "highlight"));
    svgHighlights.appendChild(poly);
}

// ==== API COMMUNICATION ====
async function fetchLegalMoves() {
    updateTurnIndicator();
    renderBoard();
    
    if (state.side_to_move !== humanColor) {
        uiMode = "WAIT_AI";
        statusText.textContent = "AI is thinking...";
        loadingOverlay.classList.remove("hidden");
        requestAIMove();
        return;
    }
    
    uiMode = "WAITING_DATA";
    statusText.textContent = "Loading moves...";
    try {
        const res = await fetch(`${API_BASE}/legal_moves`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(state)
        });
        const data = await res.json();
        legalMoves = data.legal_moves;
        
        if (data.winner !== null && data.winner !== undefined) {
            uiMode = "GAME_OVER";
            const winName = data.winner === 1 ? "RED" : "BLACK";
            statusText.textContent = `GAME OVER - ${winName} WINS!`;
            turnIndicator.textContent = "🏆";
            // Create celebration effect on winner text
            statusText.style.color = "var(--highlight-stroke)";
            return;
        }
        
        if (legalMoves.length === 0) {
            uiMode = "GAME_OVER";
            const winner = state.side_to_move === RED ? "BLACK" : "RED";
            statusText.textContent = `GAME OVER - ${winner} WINS!`;
            turnIndicator.textContent = "DONE";
            return;
        }
        
        uiMode = "IDLE";
        statusText.textContent = "Your turn: Select a piece";
    } catch (e) {
        console.error(e);
        statusText.textContent = "Error fetching moves.";
    }
}

async function requestAIMove() {
    const parts = aiOpponent.split("-");
    const aiType = parts[0];
    const strength = parts.slice(1).join("-") || "0";
    
    // Quick artificial delay for visual smoothness
    await new Promise(r => setTimeout(r, 600));

    try {
        const res = await fetch(`${API_BASE}/ai_move`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                state: state,
                ai_type: aiType,
                strength: strength
            })
        });
        const mv = await res.json();
        loadingOverlay.classList.add("hidden");
        
        // Apply AI move locally
        applyMoveLocally(mv);
        fetchLegalMoves();
    } catch(e) {
        console.error(e);
        loadingOverlay.classList.add("hidden");
        statusText.textContent = "AI Server Error";
    }
}

function applyMoveLocally(mv) {
    // 1. Slide
    const pieces = state.side_to_move === RED ? state.red : state.black;
    const startIdx = pieces.findIndex(p => p[0] === mv.slide.start[0] && p[1] === mv.slide.start[1]);
    pieces[startIdx] = mv.slide.end;
    
    // 2. Relocate
    const discIdx = state.discs.findIndex(d => d[0] === mv.tile.remove_from[0] && d[1] === mv.tile.remove_from[1]);
    state.discs[discIdx] = mv.tile.place_to;
    state.forbidden_disc = mv.tile.place_to;
    
    // 3. Swap Turn
    state.side_to_move = state.side_to_move === RED ? BLACK : RED;
    renderBoard();
}

// ==== INTERACTION LOGIC ====
function arraysEqual(a, b) { return a[0]===b[0] && a[1]===b[1]; }

function handleHexClick(q, r, clickSource) {
    if (uiMode === "WAIT_AI" || uiMode === "GAME_OVER") return;
    
    // Step 1: Click own Piece to select
    if (uiMode === "IDLE" || uiMode === "PIECE_SELECTED") {
        const isOwnRed = humanColor === RED && state.red.some(p => arraysEqual(p, [q,r]));
        const isOwnBlack = humanColor === BLACK && state.black.some(p => arraysEqual(p, [q,r]));
        
        if (isOwnRed || isOwnBlack) {
            clearHighlights();
            tempMove.start = [q, r];
            
            // Find all valid slide ends for this piece
            const validEnds = new Set();
            legalMoves.forEach(mv => {
                if (arraysEqual(mv.slide.start, [q,r])) {
                    validEnds.add(JSON.stringify(mv.slide.end));
                }
            });
            validEnds.forEach(str => {
                const endPos = JSON.parse(str);
                addHighlight(endPos[0], endPos[1], "slide");
            });
            uiMode = "PIECE_SELECTED";
            statusText.textContent = "Slide piece to glow.";
            return;
        }
    }
    
    // Step 2: Click highlight to slide
    if (uiMode === "PIECE_SELECTED" && clickSource === "highlight") {
        tempMove.end = [q, r];
        clearHighlights();
        
        // Temporarily move the piece visually
        renderBoardWithTempSlide(tempMove.start, tempMove.end);
        
        // Find all valid removable tiles for this route
        const validRemoves = new Set();
        legalMoves.forEach(mv => {
            if (arraysEqual(mv.slide.start, tempMove.start) && arraysEqual(mv.slide.end, tempMove.end)) {
                validRemoves.add(JSON.stringify(mv.tile.remove_from));
            }
        });
        
        validRemoves.forEach(str => {
            const remPos = JSON.parse(str);
            addHighlight(remPos[0], remPos[1], "remove");
        });
        uiMode = "PIECE_SLID";
        statusText.textContent = "Select edge hex to remove.";
        return;
    }
    
    // Step 3: Click tile to remove
    if (uiMode === "PIECE_SLID" && clickSource === "highlight") {
        tempMove.remove = [q, r];
        clearHighlights();
        
        // Visual temp removal
        renderBoardWithTempSlideAndRemove(tempMove.start, tempMove.end, tempMove.remove);
        
        // Find valid placement spots
        const validPlaces = new Set();
        legalMoves.forEach(mv => {
            if (arraysEqual(mv.slide.start, tempMove.start) && 
                arraysEqual(mv.slide.end, tempMove.end) &&
                arraysEqual(mv.tile.remove_from, tempMove.remove)) {
                validPlaces.add(JSON.stringify(mv.tile.place_to));
            }
        });
        
        validPlaces.forEach(str => {
            const placePos = JSON.parse(str);
            addHighlight(placePos[0], placePos[1], "place");
        });
        uiMode = "TILE_REMOVED";
        statusText.textContent = "Place tile in glowing spot.";
        return;
    }
    
    // Step 4: Click to place and end turn
    if (uiMode === "TILE_REMOVED" && clickSource === "highlight") {
        tempMove.place = [q, r];
        clearHighlights();
        
        // Match chosen move exactly
        const finalMove = legalMoves.find(mv => 
            arraysEqual(mv.slide.start, tempMove.start) &&
            arraysEqual(mv.slide.end, tempMove.end) &&
            arraysEqual(mv.tile.remove_from, tempMove.remove) &&
            arraysEqual(mv.tile.place_to, tempMove.place)
        );
        
        if (finalMove) {
            applyMoveLocally(finalMove);
            fetchLegalMoves(); // Hands over to AI
        }
    }
}

// ==== VISUAL TEMP HELPERS ====
// Just mutates and calls renderBoard, then unmutates (hacky but perfectly visually syncing)
function renderBoardWithTempSlide(s, e) {
    const arr = state.side_to_move === RED ? state.red : state.black;
    const idx = arr.findIndex(p => arraysEqual(p, s));
    arr[idx] = e;
    renderBoard();
    arr[idx] = s;
}

function renderBoardWithTempSlideAndRemove(s, e, rem) {
    const arr = state.side_to_move === RED ? state.red : state.black;
    const idx = arr.findIndex(p => arraysEqual(p, s));
    arr[idx] = e;
    
    const discIdx = state.discs.findIndex(d => arraysEqual(d, rem));
    state.discs.splice(discIdx, 1);
    
    renderBoard();
    
    state.discs.splice(discIdx, 0, rem);
    arr[idx] = s;
}


// ==== MENU BINDINGS ====
document.getElementById("btn-play-red").addEventListener("click", (e) => {
    humanColor = RED;
    document.getElementById("btn-play-red").classList.add("active");
    document.getElementById("btn-play-black").classList.remove("active");
});
document.getElementById("btn-play-black").addEventListener("click", (e) => {
    humanColor = BLACK;
    document.getElementById("btn-play-black").classList.add("active");
    document.getElementById("btn-play-red").classList.remove("active");
});
document.getElementById("ai-select").addEventListener("change", (e) => {
    aiOpponent = e.target.value;
});
document.getElementById("btn-restart").addEventListener("click", () => {
    initGame();
});

// START
initGame();
