#include <algorithm>

#include <plugin/Model.hpp>
#include <plugin.hpp>
#include <asset.hpp>
#include <system.hpp>
#include <string.hpp>
#include <tag.hpp>


namespace rack {
namespace plugin {


void Model::fromJson(json_t* rootJ) {
	assert(plugin);

	json_t* nameJ = json_object_get(rootJ, "name");
	if (nameJ)
		name = json_string_value(nameJ);
	if (name == "")
		throw Exception(string::f("No module name for slug %s", slug.c_str()));

	json_t* descriptionJ = json_object_get(rootJ, "description");
	if (descriptionJ)
		description = json_string_value(descriptionJ);

	// Tags
	tags.clear();
	json_t* tagsJ = json_object_get(rootJ, "tags");
	if (tagsJ) {
		size_t i;
		json_t* tagJ;
		json_array_foreach(tagsJ, i, tagJ) {
			std::string tag = json_string_value(tagJ);
			int tagId = tag::findId(tag);

			// Omit duplicates
			auto it = std::find(tags.begin(), tags.end(), tagId);
			if (it != tags.end())
				continue;

			if (tagId >= 0)
				tags.push_back(tagId);
		}
	}

	// manualUrl
	json_t* manualUrlJ = json_object_get(rootJ, "manualUrl");
	if (manualUrlJ)
		manualUrl = json_string_value(manualUrlJ);
}


std::string Model::getFullName() {
	assert(plugin);
	return plugin->getBrand() + " " + name;
}


std::string Model::getFactoryPresetDir() {
	return asset::plugin(plugin, system::join("presets", slug));
}


std::string Model::getUserPresetDir() {
	return asset::user(system::join("presets", plugin->slug, slug));
}




} // namespace plugin
} // namespace rack
