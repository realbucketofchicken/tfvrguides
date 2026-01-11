function Nope(){
    var audio = new Audio('nope-sfx.mp3')
    audio.play()
    var x = document.getElementById("spyscare");
    if (x.style.display === "none"){
        x.style.display = "block";
    }
}