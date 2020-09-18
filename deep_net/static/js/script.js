let canvas = document.getElementById('canv');
let ctx = canvas.getContext('2d');




function div(a, b) {
    return (a - (a % b)) / b
}


let is_mouse_down = false;
let pixel = 20;
let pixel_shape = 20;

let number_of_pixels = 28;

canvas.width = pixel * number_of_pixels;
canvas.height = pixel * number_of_pixels;

let pixels = new Array(number_of_pixels);
for (let i = 0; i < pixels.length; i++) {
    pixels[i] = new Array(number_of_pixels)
    for (let j = 0; j < pixels[i].length; j++) {
        pixels[i][j] = 0;
    }
}

draw_line = function (x1, y1, x2, y2, color = 'black') {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineJoin = 'miter';
    ctx.lineWidth = 1;
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}
clear = function () {
    ctx.clearRect(0, 0, canv.width, canv.height);
}

drawGrid = function (x, y, w, h) {
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'miter';
    ctx.strokeRect(x, y, w, h);

    for (let i = 1; i <= number_of_pixels - 1; i++) {
        draw_line(x, pixel * i, w, pixel * i);
    }
    for (let i = 1; i <= number_of_pixels - 1; i++) {
        draw_line(pixel * i, y, pixel * i, h);
    }
}

canvas.addEventListener('mousedown', function (e) {
    is_mouse_down = true;
    ctx.beginPath();
})
canvas.addEventListener('mouseup', function (e) {
    is_mouse_down = false;
})
canvas.addEventListener('mousemove', function (e) {
    if (is_mouse_down) {
        p_w = div(e.offsetX, pixel);
        p_h = div(e.offsetY, pixel);
        if (pixels[p_h][p_w] <= 250) {

            ctx.fillStyle = `rgba(0%, 0%, 0%, ${pixels[p_h][p_w] / 255.0})`;
            pixels[p_h][p_w] += 51;
            ctx.fillRect(p_w * pixel, p_h * pixel, pixel, pixel);
        }
    }
})
document.getElementById("but").addEventListener('click', function (e) {
    let xhr = new XMLHttpRequest();

    let json = JSON.stringify({
        'data': pixels,
    });

    xhr.open("POST", '/deep-net/send');
    xhr.setRequestHeader('Content-type', 'application/json; charset=utf-8');

    xhr.send(json);

    xhr.onload = function(){
        if (xhr.status != 200){
            alert('error '+ xhr.statusText);
        } else {
            alert(xhr.response);
            document.getElementById('res').textContent = "Last number: "+ xhr.response
        }
    }
})

drawGrid(0, 0, canvas.width, canvas.height);