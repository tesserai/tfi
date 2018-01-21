(function () {
    "use strict";
    function make_hover_css(top) {
        var pretty = window.innerWidth > 600;
        return (`position: absolute;
         background-color: #FFF;
         opacity: 0.95;
         top: calc(${top}px + 1.5rem);
         display: block;
         border: 1px solid rgba(0, 0, 0, 0.25);
         border-radius: ${pretty ? 3 : 0}px;
         box-shadow: 0px 2px 10px 2px rgba(0, 0, 0, 0.2);
         z-index: ${1e6};`);
    }

    function DtHoverBox(div_sel) {
        this.div = document.querySelector(div_sel);
        this.visible = false;
        this.bindDivEvents();
        DtHoverBox.box_map[div_sel] = this;
    }

    DtHoverBox.box_map = {};

    DtHoverBox.get_box = function get_box(div_sel) {
        if (div_sel in DtHoverBox.box_map) {
            return DtHoverBox.box_map[div_sel];
        } else {
            return new DtHoverBox(div_sel);
        }
    }

    DtHoverBox.prototype.show = function show(pos) {
        this.visible = true;
        this.div.setAttribute("style", make_hover_css(pos));
        for (var box_id in DtHoverBox.box_map) {
            var box = DtHoverBox.box_map[box_id];
            if (box != this) box.hide();
        }
    }

    DtHoverBox.prototype.showAtNode = function showAtNode(node) {
        var bbox = node.getBoundingClientRect();
        this.show(node.offsetTop);
    }

    DtHoverBox.prototype.hide = function hide() {
        this.visible = false;
        if (this.div) this.div.setAttribute("style", "display:none");
        if (this.timeout) clearTimeout(this.timeout);
    }

    DtHoverBox.prototype.stopTimeout = function stopTimeout() {
        if (this.timeout) clearTimeout(this.timeout);
    }

    DtHoverBox.prototype.extendTimeout = function extendTimeout(T) {
        //console.log("extend", T)
        var this_ = this;
        this.stopTimeout();
        this.timeout = setTimeout(function () { this_.hide(); }.bind(this), T);
    }

    // Bind events to a link to open this box
    DtHoverBox.prototype.bind = function bind(node) {
        if (typeof node == "string") {
            node = document.querySelector(node);
        }

        node.addEventListener("mouseover", function () {
            if (!this.visible) this.showAtNode(node);
            this.stopTimeout();
        }.bind(this));

        node.addEventListener("mouseout", function () { this.extendTimeout(250); }.bind(this));

        node.addEventListener("touchstart", function (e) {
            if (this.visible) {
                this.hide();
            } else {
                this.showAtNode(node);
            }
            // Don't trigger body touchstart event when touching link
            e.stopPropagation();
        }.bind(this));
    }

    DtHoverBox.prototype.bindDivEvents = function bindDivEvents() {
        // For mice, same behavior as hovering on links
        this.div.addEventListener("mouseover", function () {
            if (!this.visible) this.showAtNode(node);
            this.stopTimeout();
        }.bind(this));
        this.div.addEventListener("mouseout", function () { this.extendTimeout(250); }.bind(this));

        // Don't trigger body touchstart event when touching within box
        this.div.addEventListener("touchstart", function (e) { e.stopPropagation(); });
        // Close box when touching outside box
        document.body.addEventListener("touchstart", function () { this.hide(); }.bind(this));
    }

    var hover_es = document.querySelectorAll("span[data-hover-ref]");
    hover_es = [].slice.apply(hover_es);
    hover_es.forEach(function (e, n) {
        var ref_id = e.getAttribute("data-hover-ref");
        DtHoverBox.get_box(ref_id).bind(e);
    });
})();
