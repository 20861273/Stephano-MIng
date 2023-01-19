import folium
from folium.plugins import Draw

m = folium.Map()

Draw(export=True,
     filename='drawing.geojson',
     position = 'topleft').add_to(m)

mapObjectInHTML = m.get_name()

m.get_root().html.add_child(folium.Element("""
<script type="text/javascript">
  $(document).ready(function(){
    
    {map}.on("draw:created", function(e){    
        var layer = e.layer;
            feature = layer.feature = layer.feature || {}; 
            
        var title = prompt("Please provide the name of the search area", "default");

        feature.type = feature.type || "Feature";
        var props = feature.properties = feature.properties || {};
        props.Title = title;
        drawnItems.addLayer(layer);
      });    
    });    
</script>
""".replace('{map}', mapObjectInHTML)))

m.save('map.html')