```javascript
document.addEventListener("DOMContentLoaded", function () {

    console.log("AI Waste Detection System started");


    // =========================================================
    // ELEMENTS
    // =========================================================

    const imageInput =
        document.getElementById("imageInput");

    const detectForm =
        document.getElementById("detectForm");

    const selectedFile =
        document.getElementById("selectedFile");

    const loadingMessage =
        document.getElementById("loadingMessage");

    const detectButton =
        document.getElementById("detectButton");


    // =========================================================
    // IMAGE SELECT
    // =========================================================

    if (imageInput) {

        imageInput.addEventListener("change", function () {

            if (!this.files || this.files.length === 0) {
                return;
            }

            const file = this.files[0];


            // Check image type
            if (!file.type.startsWith("image/")) {

                alert("Please select a valid image file.");

                this.value = "";

                if (selectedFile) {
                    selectedFile.textContent = "";
                }

                return;
            }


            // Check file size
            const maxSize =
                10 * 1024 * 1024;

            if (file.size > maxSize) {

                alert(
                    "Image size must be less than 10 MB."
                );

                this.value = "";

                return;
            }


            // Show filename
            if (selectedFile) {

                selectedFile.textContent =
                    "Selected image: " + file.name;

            }


            // =====================================================
            // AUTOMATIC DETECTION
            // =====================================================

            setTimeout(function () {

                if (detectForm) {

                    detectForm.requestSubmit();

                }

            }, 300);

        });

    }


    // =========================================================
    // FORM SUBMIT
    // =========================================================

    if (detectForm) {

        detectForm.addEventListener(
            "submit",
            function () {

                if (!imageInput ||
                    !imageInput.files ||
                    imageInput.files.length === 0) {

                    alert(
                        "Please select an image first."
                    );

                    return;
                }


                // =================================================
                // SHOW LOADING
                // =================================================

                if (loadingMessage) {

                    loadingMessage.style.display =
                        "block";

                    loadingMessage.innerHTML =
                        "🤖 AI is detecting waste... Please wait.";

                }


                // =================================================
                // DISABLE BUTTON
                // =================================================

                if (detectButton) {

                    detectButton.disabled =
                        true;

                    detectButton.innerHTML =
                        "⏳ Detecting...";

                }


                console.log(
                    "Sending image to Flask..."
                );

            }
        );

    }


    // =========================================================
    // OPEN UPLOAD
    // =========================================================

    window.openUpload = function () {

        if (imageInput) {

            imageInput.click();

        }

    };


    // =========================================================
    // CAMERA
    // =========================================================

    window.startCamera = function () {

        const cameraContainer =
            document.getElementById(
                "cameraContainer"
            );

        const cameraFeed =
            document.getElementById(
                "cameraFeed"
            );


        if (!cameraContainer ||
            !cameraFeed) {

            window.location.href =
                "/camera";

            return;
        }


        cameraContainer.style.display =
            "block";


        cameraFeed.src =
            "/video_feed?" +
            new Date().getTime();

    };


    // =========================================================
    // STOP CAMERA
    // =========================================================

    window.stopCamera = function () {

        const cameraContainer =
            document.getElementById(
                "cameraContainer"
            );

        const cameraFeed =
            document.getElementById(
                "cameraFeed"
            );


        if (cameraFeed) {

            cameraFeed.src = "";

        }


        if (cameraContainer) {

            cameraContainer.style.display =
                "none";

        }

    };


});
```
