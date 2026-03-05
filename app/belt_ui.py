def render_belt_ui(signal):
    """
    signal: list of 5 integers representing vibration intensity (0-100)
    Generates HTML to display 5 circles with corresponding brightness.
    """
    html_parts = ['<div style="text-align:center; padding: 20px;">']
    
    # Motor order: [Left, Center-Left, Center, Center-Right, Right]
    for intensity in signal:
        # Map 0-100 to an opacity value (0.2 to 1.0)
        opacity = 0.2 + (intensity / 100.0) * 0.8
        color = f"rgba(255, 0, 0, {opacity})"
        box_shadow = f"0 0 {intensity/2}px rgba(255, 0, 0, {opacity/2})" if intensity > 0 else "none"
        
        circle_html = (
            f'<span style="display:inline-block; width:50px; height:50px; '
            f'border-radius:50%; background-color:{color}; margin:0 15px; '
            f'border:2px solid #555; box-shadow:{box_shadow};"></span>'
        )
        html_parts.append(circle_html)
        
    html_parts.append('</div>')
    return "".join(html_parts)
