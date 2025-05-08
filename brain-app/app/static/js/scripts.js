function previewFile(event) {
    console.log("File input changed");
    const file = event.target.files[0];
    const previewDiv = document.getElementById("preview");

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            console.log("File loaded:", e.target.result);
            previewDiv.innerHTML = `<img src="${e.target.result}" alt="Image preview" style="max-width: 200px; max-height: 200px;"/>`;
        };
        reader.readAsDataURL(file);
    } else {
        previewDiv.innerHTML = "<p>No file selected</p>";
    }
}