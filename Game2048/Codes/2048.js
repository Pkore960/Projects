var board;
var score = 0;
var rows = 4;
var columns = 4;
var moves = 0;
var score2 = 0;
var a=1;

var audio1 = new Audio("sound.wav");
var audio2 = new Audio("audio.wav");


window.onload = function () {
    setGame();
}

function NewGame() {

    board = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    undoboard = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            let tile = document.getElementById(r.toString() + "-" + c.toString());
            let num = board[r][c];
            updateTile(tile, num);
        }
    }
    Push();
    score = 0;
    score2 = 0;
    moves = 0;
    document.getElementById("score").innerText = score;
    document.getElementById("moves").innerText = moves;

}

function setGame() {

    board = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    undoboard = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            let tile = document.createElement("div");
            tile.id = r.toString() + "-" + c.toString();
            let num = board[r][c];
            updateTile(tile, num);
            document.getElementById("board").append(tile);
        }
    }
    Push();

}

function updateTile(tile, num) {
    tile.innerText = "";
    tile.classList.value = "";
    tile.classList.add("tile");
    if (num > 0) {
        tile.innerText = num.toString();
        if (num <= 4096) {
            tile.classList.add("x" + num.toString());
        } else {
            tile.classList.add("x8192");
        }
    }
}

document.addEventListener('keyup', (e) => {
    audio1.play();
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            undoboard[r][c] = board[r][c];
        }
    }
    score2 = score;
    a=1;
    if (e.code == "ArrowLeft") {
        slideLeft();
    }
    else if (e.code == "ArrowRight") {
        slideRight();
    }
    else if (e.code == "ArrowUp") {
        slideUp();

    }
    else if (e.code == "ArrowDown") {
        slideDown();
    }
    Push();
    moves++;
    document.getElementById("score").innerText = score;
    document.getElementById("moves").innerText = moves;
})

function filterZero(row) {
    return row.filter(num => num != 0);
}

function slide(row) {

    row = filterZero(row);
    for (let i = 0; i < row.length - 1; i++) {
        if (row[i] == row[i + 1]) {
            audio2.play();
            row[i] *= 2;
            row[i + 1] = 0;
            score += row[i];
        }
    }
    row = filterZero(row);

    while (row.length < columns) {
        row.push(0);
    }
    return row;
}

function slideLeft() {
    for (let r = 0; r < rows; r++) {
        let row = board[r];
        row = slide(row);
        board[r] = row;
        for (let c = 0; c < columns; c++) {
            let tile = document.getElementById(r.toString() + "-" + c.toString());
            let num = board[r][c];
            updateTile(tile, num);
        }
    }
}

function slideRight() {
    for (let r = 0; r < rows; r++) {
        let row = board[r];
        row.reverse();
        row = slide(row)
        board[r] = row.reverse();
        for (let c = 0; c < columns; c++) {
            let tile = document.getElementById(r.toString() + "-" + c.toString());
            let num = board[r][c];
            updateTile(tile, num);
        }
    }
}

function slideUp() {
    for (let c = 0; c < columns; c++) {
        let row = [board[0][c], board[1][c], board[2][c], board[3][c]];
        row = slide(row);
        for (let r = 0; r < rows; r++) {
            board[r][c] = row[r];
            let tile = document.getElementById(r.toString() + "-" + c.toString());
            let num = board[r][c];
            updateTile(tile, num);
        }
    }
}

function slideDown() {
    for (let c = 0; c < columns; c++) {
        let row = [board[0][c], board[1][c], board[2][c], board[3][c]]
        row.reverse();
        row = slide(row);
        row.reverse();
        for (let r = 0; r < rows; r++) {
            board[r][c] = row[r];
            let tile = document.getElementById(r.toString() + "-" + c.toString());
            let num = board[r][c];
            updateTile(tile, num);
        }
    }
}

function Push() {
    if (!hasEmptyTile()) {
        return;
    }
    let found = false;
    while (!found) {

        let r = Math.floor(Math.random() * rows);
        let c = Math.floor(Math.random() * columns);
        let a = Math.floor(Math.random() * 5);

        if (board[r][c] == 0) {
            if (a == 0) {
                board[r][c] = 4;
                let tile = document.getElementById(r.toString() + "-" + c.toString());
                tile.innerText = "4";
                tile.classList.add("x4");
                found = true;
            }
            else {
                board[r][c] = 2;
                let tile = document.getElementById(r.toString() + "-" + c.toString());
                tile.innerText = "2";
                tile.classList.add("x2");
                found = true;
            }
        }
    }
}

function hasEmptyTile() {
    let count = 0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            if (board[r][c] == 0) {
                return true;
            }
        }
    }
    return false;
}

function isEmpty() {
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            if (undoboard[r][c] != 0) {
                return false;
            }
        }
    }
    return true;
}

function Undo() {

    if (isEmpty() || a==0) {
        return;
    }
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            let tile = document.getElementById(r.toString() + "-" + c.toString());
            let num = undoboard[r][c];
            updateTile(tile, num);
        }
    }
    moves--;
    score = score2;
    document.getElementById("score").innerText = score;
    document.getElementById("moves").innerText = moves;
    a=0;
}