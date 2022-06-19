
import numpy as np
from IPython.display import IFrame
from matplotlib import pyplot as plt

TEMPLATE_VG = """
<!DOCTYPE html>
<head>

<title>PyntCloud</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
<style>
    body {{
        color: #cccccc;font-family: Monospace;
        font-size: 13px;
        text-align: center;
        background-color: #050505;
        margin: 0px;
        overflow: hidden;
    }}
    #logo_container {{
        position: absolute;
        top: 0px;
        width: 100%;
    }}
    .logo {{
        max-width: 20%;
    }}
</style>

</head>
<body>

<div>
    <img class="logo" src="">
</div>

<div id="container">
</div>

<script src="http://threejs.org/build/three.js"></script>
<script src="http://threejs.org/examples/js/controls/OrbitControls.js"></script>
<script src="http://threejs.org/examples/js/libs/stats.min.js"></script>

<script>

    var container, stats;
    var camera, scene, renderer;
    var points;

    init();
    animate();

    function init() {{

        var camera_x = {camera_x};
		var camera_y = {camera_y};
		var camera_z = {camera_z};

        var look_x = {look_x};
        var look_y = {look_y};
        var look_z = {look_z};

		var X = new Float32Array({X});
        var Y = new Float32Array({Y});
        var Z = new Float32Array({Z});

        var R = new Float32Array({R});
        var G = new Float32Array({G});
        var B = new Float32Array({B});

        var S_x = {S_x};
        var S_y = {S_y};
        var S_z = {S_z};

        var n_voxels = {n_voxels};
        var axis_size = {axis_size};

        container = document.getElementById( 'container' );

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera( 90, window.innerWidth / window.innerHeight, 0.1, 1000 );
        camera.position.x = camera_x;
        camera.position.y = camera_y;
        camera.position.z = camera_z;
        camera.up = new THREE.Vector3( 0, 1, 0 );	

        if (axis_size > 0){{
            var axisHelper = new THREE.AxesHelper( axis_size );
		    scene.add( axisHelper );
        }}

        var geometry = new THREE.BoxGeometry( S_x, S_z, S_y );

        for ( var i = 0; i < n_voxels; i ++ ) {{            
            var mesh = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial() );
            mesh.material.color.setRGB(R[i], G[i], B[i]);
            mesh.position.x = X[i];
            mesh.position.y = Y[i];
            mesh.position.z = Z[i];
            scene.add(mesh);
        }}
        
        renderer = new THREE.WebGLRenderer( {{ antialias: false }} );
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( window.innerWidth, window.innerHeight );

        controls = new THREE.OrbitControls( camera, renderer.domElement );
        controls.target.copy( new THREE.Vector3(look_x, look_y, look_z) );
        camera.lookAt( new THREE.Vector3(look_x, look_y, look_z));

        container.appendChild( renderer.domElement );

        window.addEventListener( 'resize', onWindowResize, false );
    }}

    function onWindowResize() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize( window.innerWidth, window.innerHeight );
    }}

    function animate() {{
        requestAnimationFrame( animate );
        render();
    }}

    function render() {{
        renderer.render( scene, camera );
    }}
</script>
</body>
</html>
"""

def isosceles_triangle_height(angle,mn,mx):
    """
    Assumptions:
        the base is contained in the x axis.
        the center of the coordinate system is the midpoint of the base.
        mn and mx are the (x coordinate of) left and right corners of the base.
        angle is the angle opposite to the base.
    """
    a = 1 - np.cos(angle)**2
    b = 2 * mx * mn - (np.cos(angle)**2)*(mx**2)-(np.cos(angle)**2)*(mn**2)
    c = (mx*mn)**2 - (np.cos(angle)**2)*(mx**2)*(mn**2)
    return -np.sqrt(np.max(np.roots([a,b,c])))

def plotvoxcloud(points, rgb):

    look = points.mean(0)
    axis_size = points.ptp() * 1.5

    camera_position = look.copy()
    camera_position[2] = camera_position[2] + \
        np.abs(isosceles_triangle_height(
            60*np.pi/180,
            (points-look)[:,:2].min(),
            (points-look)[:,:2].max()
        ))

    with open("plotVC.html", "w") as html:
        html.write(TEMPLATE_VG.format( 
            camera_x=camera_position[0],
            camera_y=camera_position[1],
            camera_z=camera_position[2],
            look_x=look[0],
            look_y=look[1],
            look_z=look[2],
            X=points[:,0].tolist(),
            Y=points[:,1].tolist(),
            Z=points[:,2].tolist(),
            R=rgb[:,0].tolist(),
            G=rgb[:,1].tolist(),
            B=rgb[:,2].tolist(),
            S_x=1.0,
            S_y=1.0,
            S_z=1.0,
            n_voxels=points.shape[0],
            axis_size=axis_size))

    return IFrame("plotVC.html",width=800, height=800)