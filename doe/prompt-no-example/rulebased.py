import xml.etree.ElementTree as ET
from xml.dom import minidom


def json_to_puboutages_xml(item: dict) -> str:
    """
    Rule-based conversion.
    If any exception happens, return empty string "".
    """
    try:
        NS = "http://iec.ch/TC57/2014/PubOutages#"
        P = "ns0"  # desired prefix in output

        area = str(item.get("area_name", ""))
        meters_served = int(item.get("cust_s", 0) or 0)

        cust_a = item.get("cust_a") or {}
        meters_affected = int(cust_a.get("val", 0) or 0)

        etr_val = item.get("etr", "")
        mrid = f"OUTAGE-{area}-0001"

        root = ET.Element(f"{P}:PubOutages", {f"xmlns:{P}": NS})

        outage = ET.SubElement(root, f"{P}:Outage")
        ET.SubElement(outage, f"{P}:mRID").text = mrid
        ET.SubElement(outage, f"{P}:communityDescriptor").text = area
        ET.SubElement(outage, f"{P}:cause").text = "Unknown"
        ET.SubElement(outage, f"{P}:causeKind").text = "Unknown"
        ET.SubElement(outage, f"{P}:metersAffected").text = str(meters_affected)
        ET.SubElement(outage, f"{P}:outageKind").text = "outageReported"
        ET.SubElement(outage, f"{P}:statusKind").text = "Unknown"

        ert_wrap = ET.SubElement(outage, f"{P}:EstimatedRestorationTime")
        ert = ET.SubElement(ert_wrap, f"{P}:ert")
        if str(etr_val).strip():
            ert.text = str(etr_val)  # else stays as <ns0:ert/>

        out_area = ET.SubElement(outage, f"{P}:OutageArea")
        ET.SubElement(out_area, f"{P}:metersServed").text = str(meters_served)
        ET.SubElement(out_area, f"{P}:outageAreaKind").text = "zipcode"

        inc = ET.SubElement(outage, f"{P}:Incident")
        ET.SubElement(inc, f"{P}:cause").text = "Pending Investigation"
        loc = ET.SubElement(inc, f"{P}:Location")
        ET.SubElement(loc, f"{P}:geoInfoReference").text = area
        ET.SubElement(loc, f"{P}:zoneKind").text = "zipcode"

        rough = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")
        pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
        return pretty

    except Exception:
        return ""


if __name__ == "__main__":
    sample = {
        "cust_a": {"val": 0},
        "cust_s": 158,
        "etr": "",
        "area_name": "25148",
        "index": 755,
    }
    print(json_to_puboutages_xml(sample))
